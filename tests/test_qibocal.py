import signal
from copy import deepcopy
from functools import wraps

import numpy as np
import pytest

pytest.importorskip("qibocal")

from examples.calibration import (  # noqa: E402
    PROTOCOLS,
    acquire,
    apply_update,
    calibration_platform,
    fit,
    report,
)


@pytest.mark.parametrize("name", list(PROTOCOLS))
def test_qibocal_phases_on_coupled_qutrits(name):
    session = acquire(name, points=17, nshots=4096)
    assert not session.platform.is_connected
    assert session.platform.config("hamiltonian").couplings[0].coupling == 12e6
    assert set(session.data.qubits) == {0, 1}
    raw = deepcopy(session.data.data)
    for records in raw.values():
        assert len(records) >= 17
        assert all(np.all(np.isfinite(records[field])) for field in records.dtype.names)
    fitted = fit(session)
    assert all(target in fitted for target in session.targets)
    for key, records in raw.items():
        np.testing.assert_array_equal(session.data.data[key], records)
    for target, figures, table in report(session, fitted):
        assert target in session.targets
        assert figures
        assert isinstance(table, str)
        for figure in figures:
            assert figure.to_json()
    apply_update(session, fitted)
    if name == "rabi_amplitude":
        for target in session.targets:
            amplitude = session.platform.natives.single_qubit[target].RX[0][1].amplitude
            assert amplitude == fitted.amplitude[target][0]
            assert amplitude == pytest.approx(0.1265, abs=0.005)
    if name == "flux_amplitude_frequency":
        for target in session.targets:
            assert (
                session.platform.calibration.single_qubits[
                    target
                ].qubit.flux_coefficients
                is not None
            )


@pytest.mark.parametrize("name,field", [("t1", "t1"), ("t2", "t2")])
def test_isolated_full_window_recovers_coherence(name, field):
    session = acquire(
        name,
        targets=(0,),
        coupled=False,
        points=17,
        nshots=16384,
        full_window=True,
    )
    fitted = fit(session)
    expected = getattr(session.platform.config("hamiltonian").qubits[0], field)
    assert getattr(fitted, field)[0][0] == pytest.approx(expected, rel=0.15)


def test_fit_error_floor_does_not_change_raw_data():
    session = acquire("rabi_amplitude", targets=(0,), points=17, nshots=4096)
    session.data[0].error[0] = 0
    fit(session)
    assert session.data[0].error[0] == 0
    assert session.fit_data[0].error[0] == 1 / session.parameters.nshots
    assert session.regularized_points >= 1


def test_notebook_platform_preserves_signal_handlers():
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    calibration_platform()
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers


def test_explicit_qibocal_parameter_overrides():
    session = acquire(
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
    assert session.parameters.nshots == 256
    assert session.parameters.pulse_length == 80
    assert len(session.data[0]) == 5


def test_acquisition_preserves_supplied_session_and_disconnects_on_failure(monkeypatch):
    platform = calibration_platform(coupled=False)
    platform.update({"configs.0/drive.frequency": 4.8e9})
    session = acquire("rabi_amplitude", targets=(0,), platform=platform)
    assert session.platform is platform
    assert platform.config("0/drive").frequency == 4.8e9

    @wraps(PROTOCOLS["rabi_amplitude"].acquisition)
    def failure(*args):
        raise RuntimeError("acquisition failed")

    monkeypatch.setattr(PROTOCOLS["rabi_amplitude"], "acquisition", failure)
    with pytest.raises(RuntimeError, match="acquisition failed"):
        acquire("rabi_amplitude", targets=(0,), platform=platform)
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
        acquire(**({"name": "rabi_amplitude"} | kwargs))
