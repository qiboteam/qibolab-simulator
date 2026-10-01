import json
from copy import deepcopy
from functools import wraps
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qibocal")
mo = pytest.importorskip("marimo")

from qibocal import Executor, create_calibration_platform, protocols  # noqa: E402
from qibocal.auto.operation import BoundProtocol, Completed  # noqa: E402

from notebooks.calibration import (  # noqa: E402
    acquire_protocol,
    app,
    bind_protocol,
    fit_protocol,
    run_protocol,
)
from notebooks.calibration_helpers import (  # noqa: E402
    prepare_fit_data,
    require_fit_results,
)

SWEEPS = {
    "rabi_amplitude": {"min_amp": 0.01, "max_amp": 0.28, "step_amp": 0.016},
    "rabi_length": {
        "pulse_duration_start": 4,
        "pulse_duration_end": 72,
        "pulse_duration_step": 4,
    },
    "ramsey": {"delay": [40, 4290, 250], "detuning": 1e6},
    "t1": {
        "delay_before_readout_start": 235,
        "delay_before_readout_end": 4230,
        "delay_before_readout_step": 235,
    },
    "t2": {
        "delay_between_pulses_start": 235,
        "delay_between_pulses_end": 4230,
        "delay_between_pulses_step": 235,
    },
    "spin_echo": {
        "delay_between_pulses_start": 235,
        "delay_between_pulses_end": 4230,
        "delay_between_pulses_step": 235,
    },
    "qubit_spectroscopy": {
        "freq_width": 30_000_000,
        "freq_step": 30_000_000 // 17,
        "drive_duration": 100,
        "drive_amplitude": 0.015,
    },
    "flux_amplitude_frequency": {
        "amplitude_min": 0,
        "amplitude_max": 0.04,
        "amplitude_step": 0.04 / 17,
        "duration": 100,
    },
}


@pytest.fixture(autouse=True)
def platform_path(monkeypatch):
    monkeypatch.setenv(
        "QIBOLAB_PLATFORMS",
        str(Path(__file__).resolve().parents[1] / "examples" / "platforms"),
    )


@pytest.fixture
def isolated_platform():
    platform = create_calibration_platform("split_transmons")
    platform.update({"configs.hamiltonian.couplings": []})
    return platform


def test_initial_calibration_matches_simulator_model():
    platform = create_calibration_platform("split_transmons")
    for target, model in platform.config("hamiltonian").qubits.items():
        calibration = platform.calibration.single_qubits[target]
        assert calibration.qubit.frequency_01 == model.frequency
        assert calibration.qubit.frequency_12 == model.frequency + model.anharmonicity
        assert calibration.qubit.maximum_frequency == model.frequency
        assert calibration.qubit.sweetspot == model.sweetspot
        assert calibration.qubit.asymmetry == model.asymmetry
        assert calibration.t1 == (model.t1, None)
        assert calibration.t2 == (model.t2, None)


@pytest.mark.parametrize("name", SWEEPS)
def test_qibocal_notebook_phases_on_coupled_qutrits(monkeypatch, name):
    monkeypatch.setattr(
        mo,
        "cli_args",
        lambda: {"protocol": name, "parameters": json.dumps(SWEEPS[name])},
    )
    _, definitions = app.run()
    platform = definitions["platform"]
    executor, bound = definitions["executor"], definitions["bound"]
    data, fitted = definitions["data"], definitions["fitted"]
    assert isinstance(executor, Executor)
    assert isinstance(bound, BoundProtocol)
    assert bound.protocol is getattr(protocols, name)
    assert executor.platform is platform
    assert not platform.is_connected
    assert platform.config("hamiltonian").couplings[0].coupling == 12e6
    assert set(data.qubits) == {0, 1}
    for records in data.data.values():
        assert len(records) >= 17
        if records.dtype.names is None:
            assert np.all(np.isfinite(records))
        else:
            assert all(
                np.all(np.isfinite(records[field])) for field in records.dtype.names
            )
    assert all(target in fitted for target in (0, 1))
    completed = definitions["completed"]
    assert isinstance(completed, Completed)
    assert completed.success
    assert completed.results is not None
    assert set(completed.data.qubits) == {0, 1}
    assert set(completed.reports) == {0, 1}
    assert definitions["full_platform"] is not platform
    assert not definitions["full_platform"].is_connected
    reports = executor.report(data, fitted, bound)
    assert set(reports) == {0, 1}
    for figures, table in reports.values():
        assert figures
        assert isinstance(table, str)
        for figure in figures:
            assert figure.to_json()
    if name == "rabi_amplitude":
        assert platform.natives.single_qubit[0].RX[0][1].amplitude == 0.1265
    executor.update(fitted, bound)
    if name == "rabi_amplitude":
        for target in (0, 1):
            amplitude = platform.natives.single_qubit[target].RX[0][1].amplitude
            assert amplitude == fitted.amplitude[target][0]
            assert amplitude == pytest.approx(0.1265, abs=0.005)
    if name == "flux_amplitude_frequency":
        for target in (0, 1):
            assert (
                platform.calibration.single_qubits[target].qubit.flux_coefficients
                is not None
            )


@pytest.mark.parametrize("name,field", [("t1", "t1"), ("t2", "t2")])
def test_isolated_full_window_recovers_coherence(
    monkeypatch, isolated_platform, name, field
):
    platform = isolated_platform
    window = 2 * getattr(platform.config("hamiltonian").qubits[0], field)
    start, end, step = SWEEPS[name]
    parameters = {
        start: int(window / 17),
        end: int(window / 17) * 18,
        step: int(window / 17),
    }
    monkeypatch.setattr(
        mo,
        "cli_args",
        lambda: {
            "protocol": name,
            "parameters": json.dumps(parameters | {"nshots": 16384}),
        },
    )
    _, definitions = app.run(
        defs={"platform": platform, "coupled": False},
    )
    expected = getattr(platform.config("hamiltonian").qubits[0], field)
    assert getattr(definitions["fitted"], field)[0][0] == pytest.approx(
        expected, rel=0.15
    )


def test_notebook_fit_error_floor_does_not_change_raw_data(
    monkeypatch, isolated_platform
):
    platform = isolated_platform
    bound = protocols.rabi_amplitude(**SWEEPS["rabi_amplitude"], nshots=4096)
    executor = Executor(platform, targets=[0], update=False)
    try:
        platform.connect()
        data = executor.acquire(bound)
    finally:
        platform.disconnect()
    data[0].error[0] = 0
    raw = deepcopy(data.data)
    original_fit = bound.protocol.fit

    def capture_fit(fit_data):
        assert fit_data is not data
        assert fit_data[0].error[0] == 1 / bound.parameters.nshots
        return original_fit(fit_data)

    monkeypatch.setattr(bound.protocol, "fit", capture_fit)
    fit_protocol.run(
        bound=bound,
        data=data,
        executor=executor,
        targets=[0],
    )
    np.testing.assert_array_equal(data[0], raw[0])


def test_fit_data_without_error_floor_is_an_unmodified_copy():
    data = protocols.rabi_amplitude.data_type(rx90=False)
    data.data[0] = np.rec.fromarrays([[0.0, 0.1]], names=["error"])
    fit_data = prepare_fit_data(data, 1000, error_floor=False)
    np.testing.assert_array_equal(fit_data[0], data[0])
    fit_data[0].error[0] = 1.0
    assert data[0].error[0] == 0.0


@pytest.mark.parametrize("results", [None, {}, {0: object()}])
def test_missing_fit_results_are_rejected(results):
    with pytest.raises(ValueError, match="Qibocal fit has no results"):
        require_fit_results(results, [0, 1])


@pytest.mark.parametrize(
    "parameters, message",
    [
        ([], "must be a dictionary"),
        ({"nshots": True}, "positive integer"),
        ({"nshots": 0}, "positive integer"),
        ({"nshots": 1.5}, "positive integer"),
    ],
)
def test_notebook_rejects_invalid_acquisition_parameters(parameters, message):
    with pytest.raises(ValueError, match=message):
        bind_protocol.run(routine=protocols.rabi_amplitude, parameters=parameters)


@pytest.mark.parametrize("targets", [[], [0, 0], [2]])
def test_notebook_rejects_invalid_target_selection(targets):
    platform = create_calibration_platform("split_transmons")
    with pytest.raises(ValueError, match="distinct qutrit targets"):
        bind_protocol.run(
            platform=platform,
            routine=protocols.rabi_amplitude,
            parameters={},
            targets=targets,
        )
    assert not platform.is_connected


def test_notebook_code_settings_and_optional_update(monkeypatch, isolated_platform):
    def unexpected_control(*args, **kwargs):
        raise AssertionError("Notebook input controls must be configured through code")

    for control in (
        "checkbox",
        "dropdown",
        "dictionary",
        "multiselect",
        "number",
        "text_area",
        "run_button",
        "button",
    ):
        monkeypatch.setattr(mo.ui, control, unexpected_control)
    _, definitions = app.run(
        defs={
            "platform": isolated_platform,
            "coupled": False,
            "targets": [0],
            "nshots": 4096,
        }
    )
    platform, fitted = definitions["platform"], definitions["fitted"]
    assert not platform.config("hamiltonian").couplings
    assert set(definitions["data"].qubits) == {0}
    assert not definitions["apply_update"]
    assert not definitions["full_update"]
    assert 0 in fitted
    assert platform.natives.single_qubit[0].RX[0][1].amplitude == 0.1265
    assert platform.natives.single_qubit[1].RX[0][1].amplitude == 0.1266


def test_explicit_qibocal_parameters_and_selected_target_update(isolated_platform):
    platform = isolated_platform
    executor = Executor(platform, targets=[0, 1], update=False)
    bound = protocols.rabi_amplitude(
        min_amp=0.05, max_amp=0.1, step_amp=0.01, pulse_length=80, nshots=256
    )
    assert bound.protocol is protocols.rabi_amplitude
    assert bound.parameters.nshots == 256
    assert bound.parameters.pulse_length == 80
    assert (
        protocols.rabi_amplitude(pars=bound.parameters).parameters is bound.parameters
    )
    try:
        platform.connect()
        data = executor.acquire(bound, targets=[0])
    finally:
        platform.disconnect()
    assert set(data.qubits) == {0}
    assert len(data[0]) == 5
    original = platform.natives.single_qubit[1].RX[0][1].amplitude
    fitted = executor.fit(data, bound)
    assert 0 in fitted
    executor.update(fitted, bound, targets=[0])
    assert platform.natives.single_qubit[0].RX[0][1].amplitude == fitted.amplitude[0][0]
    assert platform.natives.single_qubit[1].RX[0][1].amplitude == original


def test_notebook_acquisition_preserves_session_and_disconnects_on_failure(
    monkeypatch, isolated_platform
):
    platform = isolated_platform
    platform.update({"configs.0/drive.frequency": 4.8e9})
    _, definitions = bind_protocol.run(
        platform=platform,
        routine=protocols.rabi_amplitude,
        parameters=SWEEPS["rabi_amplitude"],
    )
    acquire_protocol.run(
        platform=platform,
        executor=definitions["executor"],
        bound=definitions["bound"],
    )
    assert definitions["executor"].platform is platform
    assert platform.config("0/drive").frequency == 4.8e9
    assert not platform.is_connected

    @wraps(protocols.rabi_amplitude.acquisition)
    def failure(*args, **kwargs):
        raise RuntimeError("acquisition failed")

    monkeypatch.setattr(protocols.rabi_amplitude, "acquisition", failure)
    with pytest.raises(RuntimeError, match="acquisition failed"):
        acquire_protocol.run(
            platform=platform,
            executor=definitions["executor"],
            bound=definitions["bound"],
        )
    assert not platform.is_connected


def test_binding_is_lazy_and_full_execution_updates_platform(isolated_platform):
    platform = isolated_platform
    original = platform.natives.single_qubit[0].RX[0][1].amplitude
    bound = protocols.rabi_amplitude(**SWEEPS["rabi_amplitude"], nshots=4096)
    executor = Executor(platform, targets=[0])
    assert not platform.is_connected
    assert platform.natives.single_qubit[0].RX[0][1].amplitude == original
    try:
        platform.connect()
        completed = executor(bound)
    finally:
        platform.disconnect()
    assert isinstance(completed, Completed)
    assert completed.success
    assert completed.results.amplitude[0][0] == pytest.approx(0.1265, abs=0.005)
    assert (
        platform.natives.single_qubit[0].RX[0][1].amplitude
        == completed.results.amplitude[0][0]
    )
    assert set(completed.reports) == {0}
    assert not platform.is_connected


def test_skip_fit_does_not_report_or_update(isolated_platform):
    platform = isolated_platform
    original = platform.natives.single_qubit[0].RX[0][1].amplitude
    bound = protocols.rabi_amplitude(**SWEEPS["rabi_amplitude"], nshots=4096)
    executor = Executor(platform, targets=[0])
    try:
        platform.connect()
        completed = executor(bound, skip_fit=True)
    finally:
        platform.disconnect()
    assert completed.results is None
    assert set(completed.data.qubits) == {0}
    assert completed.reports is None
    assert platform.natives.single_qubit[0].RX[0][1].amplitude == original


def test_notebook_failed_fit_cannot_update_platform(monkeypatch):
    monkeypatch.setattr(mo, "cli_args", lambda: {"protocol": "rabi_amplitude"})
    _, definitions = app.run()
    fitted = definitions["fitted"]
    fitted.amplitude.clear()
    monkeypatch.setattr(protocols.rabi_amplitude, "fit", lambda data: fitted)
    with pytest.raises(ValueError, match="no results"):
        fit_protocol.run(
            **{
                name: definitions[name]
                for name in ("bound", "data", "executor", "targets")
            }
        )


def test_notebook_full_operation_disconnects_on_failure(monkeypatch, isolated_platform):
    def failure(self, bound):
        raise RuntimeError("full operation failed")

    monkeypatch.setattr(Executor, "__call__", failure)
    with pytest.raises(RuntimeError, match="full operation failed"):
        run_protocol.run(
            bound=protocols.rabi_amplitude(**SWEEPS["rabi_amplitude"], nshots=4096),
            create_calibration_platform=lambda name: isolated_platform,
            targets=[0],
        )
    assert not isolated_platform.is_connected
