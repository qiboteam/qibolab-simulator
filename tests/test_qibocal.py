import signal
from copy import deepcopy
from dataclasses import replace
from functools import wraps

import numpy as np
import pytest

pytest.importorskip("qibocal")

from qibocal import Executor  # noqa: E402
from qibocal.auto.operation import BoundProtocol, Completed  # noqa: E402

from examples.calibration import (  # noqa: E402
    PROTOCOLS,
    bind,
    calibration_platform,
)


@pytest.mark.parametrize("name", list(PROTOCOLS))
def test_qibocal_phases_on_coupled_qutrits(name):
    platform = calibration_platform()
    reports = []
    executor, bound = bind(
        name, platform=platform, points=17, nshots=4096, reports=reports
    )
    assert isinstance(executor, Executor)
    assert isinstance(bound, BoundProtocol)
    data = executor.acquire(bound)
    assert not platform.is_connected
    assert platform.config("hamiltonian").couplings[0].coupling == 12e6
    assert set(data.qubits) == {0, 1}
    raw = deepcopy(data.data)
    for records in raw.values():
        assert len(records) >= 17
        if records.dtype.names is None:
            assert np.all(np.isfinite(records))
        else:
            assert all(
                np.all(np.isfinite(records[field])) for field in records.dtype.names
            )
    fitted = executor.fit(data, bound)
    assert all(target in fitted for target in (0, 1))
    for key, records in raw.items():
        np.testing.assert_array_equal(data.data[key], records)
    executor.report(data, fitted, bound)
    for target, figures, table in reports:
        assert target in (0, 1)
        assert figures
        assert isinstance(table, str)
        for figure in figures:
            assert figure.to_json()
    assert len(reports) == 2
    executor.update(fitted)
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
def test_isolated_full_window_recovers_coherence(name, field):
    platform = calibration_platform(coupled=False)
    executor, bound = bind(
        name,
        targets=(0,),
        platform=platform,
        points=17,
        nshots=16384,
        full_window=True,
    )
    data = executor.acquire(bound)
    fitted = executor.fit(data, bound)
    expected = getattr(platform.config("hamiltonian").qubits[0], field)
    assert getattr(fitted, field)[0][0] == pytest.approx(expected, rel=0.15)


@pytest.mark.parametrize("error_floor", [True, False])
def test_fit_error_floor_does_not_change_raw_data(monkeypatch, error_floor):
    executor, bound = bind("rabi_amplitude", targets=(0,), points=17, nshots=4096)
    data = executor.acquire(bound)
    data[0].error[0] = 0
    routine = PROTOCOLS["rabi_amplitude"]
    original_fit = routine.fit

    def capture_fit(fit_data):
        assert fit_data is not data
        assert fit_data[0].error[0] == (
            1 / bound.parameters.nshots if error_floor else 0
        )
        # Keep the actual regression fit well-conditioned in both branches.
        fit_data[0].error[0] = 1 / bound.parameters.nshots
        return original_fit(fit_data)

    monkeypatch.setattr(routine, "fit", capture_fit)
    executor.fit(data, replace(bound, fitpars=error_floor))
    assert data[0].error[0] == 0


def test_notebook_platform_preserves_signal_handlers():
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    calibration_platform()
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers


def test_explicit_qibocal_parameter_overrides():
    executor, bound = bind(
        "rabi_amplitude",
        targets=(0,),
        overrides={
            "min_amp": 0.05,
            "max_amp": 0.1,
            "step_amp": 0.01,
            "pulse_length": 80,
            "nshots": 256,
        },
    )
    data = executor.acquire(bound)
    assert bound.parameters.nshots == 256
    assert bound.parameters.pulse_length == 80
    assert len(data[0]) == 5


def test_acquisition_preserves_supplied_session_and_disconnects_on_failure(monkeypatch):
    platform = calibration_platform(coupled=False)
    platform.update({"configs.0/drive.frequency": 4.8e9})
    executor, bound = bind("rabi_amplitude", targets=(0,), platform=platform)
    assert executor.platform.platform is platform
    executor.acquire(bound)
    assert platform.config("0/drive").frequency == 4.8e9

    @wraps(PROTOCOLS["rabi_amplitude"].acquisition)
    def failure(*args):
        raise RuntimeError("acquisition failed")

    monkeypatch.setattr(PROTOCOLS["rabi_amplitude"], "acquisition", failure)
    with pytest.raises(RuntimeError, match="acquisition failed"):
        executor.acquire(bound)
    assert not platform.is_connected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "unknown"},
        {"targets": ()},
        {"targets": (2,)},
        {"targets": (0, 0)},
        {"points": 8},
        {"nshots": 127},
    ],
)
def test_invalid_protocol_inputs(kwargs):
    with pytest.raises(ValueError):
        bind(**({"name": "rabi_amplitude"} | kwargs))


def test_binding_is_lazy_and_full_execution_updates_platform():
    platform = calibration_platform(coupled=False)
    original = platform.natives.single_qubit[0].RX[0][1].amplitude
    reports = []
    executor, bound = bind(
        "rabi_amplitude", platform=platform, targets=(0,), nshots=4096, reports=reports
    )
    assert not platform.is_connected
    assert platform.natives.single_qubit[0].RX[0][1].amplitude == original
    completed = executor(bound)
    assert isinstance(completed, Completed)
    assert completed.success
    assert completed.results.amplitude[0][0] == pytest.approx(0.1265, abs=0.005)
    assert (
        platform.natives.single_qubit[0].RX[0][1].amplitude
        == completed.results.amplitude[0][0]
    )
    assert len(reports) == 1
    assert not platform.is_connected


def test_skip_fit_does_not_report_or_update():
    platform = calibration_platform(coupled=False)
    original = platform.natives.single_qubit[0].RX[0][1].amplitude
    reports = []
    executor, bound = bind(
        "rabi_amplitude", platform=platform, targets=(0,), reports=reports
    )
    completed = executor(bound, skip_fit=True)
    assert completed.results is None
    assert set(completed.data.qubits) == {0}
    assert reports == []
    assert platform.natives.single_qubit[0].RX[0][1].amplitude == original


def test_failed_fit_cannot_update_platform(monkeypatch):
    executor, bound = bind("rabi_amplitude", targets=(0,))
    data = executor.acquire(bound)
    fitted = executor.fit(data, bound)
    fitted.amplitude.clear()
    monkeypatch.setattr(PROTOCOLS["rabi_amplitude"], "fit", lambda data: fitted)
    with pytest.raises(ValueError, match="no results"):
        executor.fit(data, bound)
    with pytest.raises(ValueError, match="no results"):
        executor.update(fitted)
