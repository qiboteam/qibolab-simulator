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
    create_executor,
    display_completed_reports,
    fit_protocol,
    run_protocol,
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
def notebook_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "QIBOLAB_PLATFORMS",
        str(Path(__file__).resolve().parents[1] / "examples" / "platforms"),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mo, "cli_args", lambda: {})


@pytest.fixture
def isolated_platform():
    platform = create_calibration_platform("split_transmons")
    platform.update({"configs.hamiltonian.couplings": []})
    return platform


@pytest.fixture
def isolated_executor(isolated_platform, tmp_path):
    return Executor.create(
        platform=isolated_platform,
        path=tmp_path / "report",
        targets=[0, 1],
        update=False,
    )


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
    _, definitions = create_executor.run()
    executor = definitions["executor"]
    platform = executor.platform
    _, definitions = bind_protocol.run()
    bound = definitions["bound"]
    assert isinstance(bound, BoundProtocol)
    assert bound.protocol is getattr(protocols, name)
    assert platform.config("hamiltonian").couplings[0].coupling == 12e6
    _, definitions = acquire_protocol.run(bound=bound, executor=executor)
    data = definitions["data"]
    assert not platform.is_connected
    assert set(data.qubits) == {0, 1}
    for records in data.data.values():
        assert len(records) >= 17
        if records.dtype.names is None:
            assert np.all(np.isfinite(records))
        else:
            assert all(
                np.all(np.isfinite(records[field])) for field in records.dtype.names
            )
    _, definitions = fit_protocol.run(bound=bound, data=data, executor=executor)
    fitted = definitions["fitted"]
    assert all(target in fitted for target in (0, 1))
    reports = executor.report(data, fitted, bound, targets=[0, 1])
    assert set(reports) == {0, 1}
    for figures, table in reports.values():
        assert figures
        assert isinstance(table, str)
        for figure in figures:
            assert figure.to_json()
    if name == "rabi_amplitude":
        assert platform.natives.single_qubit[0].RX[0][1].amplitude == 0.1265
    executor.update(fitted, bound, targets=[0, 1])
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
def test_isolated_full_window_recovers_coherence(isolated_executor, name, field):
    executor = isolated_executor
    platform = executor.platform
    window = 2 * getattr(platform.config("hamiltonian").qubits[0], field)
    start, end, step = SWEEPS[name]
    bound = getattr(protocols, name)(
        **{
            start: int(window / 17),
            end: int(window / 17) * 18,
            step: int(window / 17),
        }
    )
    with executor:
        data = executor.acquire(bound, targets=[0], nshots=16384)
    fitted = executor.fit(data, bound, targets=[0])
    expected = getattr(platform.config("hamiltonian").qubits[0], field)
    assert getattr(fitted, field)[0][0] == pytest.approx(expected, rel=0.15)


def test_notebook_passes_raw_data_directly_to_fit(monkeypatch, isolated_executor):
    bound = protocols.rabi_amplitude(**SWEEPS["rabi_amplitude"])
    _, definitions = acquire_protocol.run(bound=bound, executor=isolated_executor)
    data = definitions["data"]
    raw = deepcopy(data.data)
    original_fit = bound.protocol.fit

    def capture_fit(fit_data):
        assert fit_data is data
        return original_fit(fit_data)

    monkeypatch.setattr(bound.protocol, "fit", capture_fit)
    fit_protocol.run(bound=bound, data=data, executor=isolated_executor)
    for target, records in raw.items():
        np.testing.assert_array_equal(data[target], records)


def test_notebook_default_ramsey_binding():
    _, definitions = bind_protocol.run()
    bound = definitions["bound"]
    assert bound.protocol is protocols.ramsey
    assert bound.parameters.delay == (10, 1000, 100)


def test_notebook_executor_creation_handles_first_run_and_rerun():
    _, definitions = create_executor.run()
    executor = definitions["executor"]
    assert executor.path == Path("var/report")
    assert not executor.platform.is_connected
    with executor:
        pass
    assert (executor.path / "meta.json").is_file()
    _, definitions = create_executor.run()
    replacement = definitions["executor"]
    assert replacement.platform is not executor.platform
    assert not replacement.path.exists()
    with replacement:
        pass
    assert not replacement.platform.is_connected


def test_notebook_code_settings_and_headless_override(monkeypatch):
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
    monkeypatch.setattr(mo, "cli_args", lambda: {"protocol": "rabi_amplitude"})
    _, definitions = app.run()
    executor = definitions["executor"]
    assert not executor.platform.is_connected
    assert set(definitions["data"].qubits) == {0, 1}
    assert set(definitions["completed"].data.qubits) == {0}
    assert definitions["completed"].data is not definitions["data"]
    assert definitions["bound"].protocol is protocols.rabi_amplitude
    assert not definitions["apply_update"]
    assert executor.platform.natives.single_qubit[0].RX[0][1].amplitude == 0.1265
    assert executor.platform.natives.single_qubit[1].RX[0][1].amplitude == 0.1266
    assert (executor.path / "meta.json").is_file()


def test_explicit_qibocal_parameters_and_selected_target_update(isolated_executor):
    executor = isolated_executor
    platform = executor.platform
    bound = protocols.rabi_amplitude(
        min_amp=0.05, max_amp=0.1, step_amp=0.01, pulse_length=80
    )
    assert bound.parameters.pulse_length == 80
    assert (
        protocols.rabi_amplitude(pars=bound.parameters).parameters is bound.parameters
    )
    with executor:
        data = executor.acquire(bound, targets=[0], nshots=256)
    assert set(data.qubits) == {0}
    assert len(data[0]) == 5
    assert bound.parameters.nshots is None
    original = platform.natives.single_qubit[1].RX[0][1].amplitude
    fitted = executor.fit(data, bound, targets=[0])
    assert 0 in fitted
    executor.update(fitted, bound, targets=[0])
    assert platform.natives.single_qubit[0].RX[0][1].amplitude == fitted.amplitude[0][0]
    assert platform.natives.single_qubit[1].RX[0][1].amplitude == original


def test_notebook_acquisition_preserves_session_and_disconnects_on_failure(
    monkeypatch, isolated_executor
):
    executor = isolated_executor
    platform = executor.platform
    platform.update({"configs.0/drive.frequency": 4.8e9})
    bound = protocols.rabi_amplitude(**SWEEPS["rabi_amplitude"])
    acquire_protocol.run(bound=bound, executor=executor)
    assert platform.config("0/drive").frequency == 4.8e9
    assert not platform.is_connected

    @wraps(protocols.rabi_amplitude.acquisition)
    def failure(*args, **kwargs):
        raise RuntimeError("acquisition failed")

    monkeypatch.setattr(protocols.rabi_amplitude, "acquisition", failure)
    with pytest.raises(RuntimeError, match="acquisition failed"):
        acquire_protocol.run(bound=bound, executor=executor)
    assert not platform.is_connected


def test_binding_is_lazy_and_full_execution_updates_platform(
    isolated_platform, tmp_path
):
    platform = isolated_platform
    original = platform.natives.single_qubit[0].RX[0][1].amplitude
    bound = protocols.rabi_amplitude(**SWEEPS["rabi_amplitude"])
    executor = Executor.create(platform=platform, path=tmp_path / "report", targets=[0])
    assert not platform.is_connected
    assert platform.natives.single_qubit[0].RX[0][1].amplitude == original
    with executor:
        completed = executor(bound, nshots=4096)
    assert isinstance(completed, Completed)
    assert completed.success
    assert completed.results.amplitude[0][0] == pytest.approx(0.1265, abs=0.005)
    assert (
        platform.natives.single_qubit[0].RX[0][1].amplitude
        == completed.results.amplitude[0][0]
    )
    assert set(completed.reports) == {0}
    assert not platform.is_connected


def test_skip_fit_does_not_report_or_update(isolated_executor):
    executor = isolated_executor
    platform = executor.platform
    original = platform.natives.single_qubit[0].RX[0][1].amplitude
    bound = protocols.rabi_amplitude(**SWEEPS["rabi_amplitude"])
    with executor:
        completed = executor(bound, targets=[0], nshots=4096, skip_fit=True)
    assert completed.results is None
    assert set(completed.data.qubits) == {0}
    assert completed.reports is None
    assert platform.natives.single_qubit[0].RX[0][1].amplitude == original


def test_notebook_full_operation_disconnects_on_failure(monkeypatch, isolated_executor):
    def failure(self, bound, **kwargs):
        raise RuntimeError("full operation failed")

    monkeypatch.setattr(Executor, "__call__", failure)
    with pytest.raises(RuntimeError, match="full operation failed"):
        run_protocol.run(
            bound=protocols.rabi_amplitude(**SWEEPS["rabi_amplitude"]),
            executor=isolated_executor,
        )
    assert not isolated_executor.platform.is_connected


def test_notebook_full_operation_generates_and_displays_reports(
    monkeypatch, isolated_executor
):
    executor = isolated_executor
    platform = executor.platform
    original = platform.natives.single_qubit[0].RX[0][1].amplitude
    bound = protocols.rabi_amplitude(**SWEEPS["rabi_amplitude"], nshots=4096)
    monkeypatch.setattr(mo, "cli_args", lambda: {"protocol": "rabi_amplitude"})
    _, definitions = run_protocol.run(bound=bound, executor=executor)
    completed = definitions["completed"]
    assert set(completed.reports) == {0}
    figures, table = completed.reports[0]
    assert figures
    assert table
    output, _ = display_completed_reports.run(completed=completed)
    assert isinstance(output, mo.Html)
    assert "Target 0" in output.text
    assert not platform.is_connected
    assert platform.natives.single_qubit[0].RX[0][1].amplitude == original


def test_notebook_full_operation_defaults_to_qubit_flux(monkeypatch, isolated_executor):
    bound = protocols.rabi_amplitude(**SWEEPS["rabi_amplitude"])
    completed = Completed(data=None, results=None, success=True, reports={})

    def capture_call(self, operation, **kwargs):
        assert self is isolated_executor
        assert self.platform.is_connected
        assert operation.protocol is protocols.qubit_flux
        assert operation.parameters.frequency == ("linspace", 4.78e9, 4.83e9, 40)
        assert operation.parameters.bias == ("linspace", 0.35, 0.46, 30)
        assert operation.parameters.drive_duration == 1000
        assert kwargs == {"nshots": 4096, "targets": [0]}
        return completed

    monkeypatch.setattr(Executor, "__call__", capture_call)
    _, definitions = run_protocol.run(bound=bound, executor=isolated_executor)
    assert definitions["completed"] is completed
    assert not isolated_executor.platform.is_connected
