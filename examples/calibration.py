"""Explicit, in-memory Qibocal phases shared by the notebook and regressions."""

import signal
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from qibocal import protocols
from qibocal.auto.operation import Data, Parameters, Protocol, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.calibration.calibration import Calibration, QubitCalibration

from examples.platforms.split_transmons import create_platform

PROTOCOLS: dict[str, Protocol] = {
    name: getattr(protocols, name)
    for name in (
        "rabi_amplitude",
        "rabi_length",
        "ramsey",
        "t1",
        "t2",
        "spin_echo",
        "qubit_spectroscopy",
        "flux_amplitude_frequency",
    )
}


def calibration_platform(*, coupled: bool = True) -> CalibrationPlatform:
    """Wrap a fresh example without Qibocal's filesystem-based platform lookup."""
    # Platform constructors install handlers that interfere with marimo shutdown.
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        platform = create_platform()
        if not coupled:
            platform.update({"configs.hamiltonian.couplings": []})
        model = platform.config("hamiltonian")
        calibration = Calibration(
            single_qubits={
                target: QubitCalibration(
                    qubit={
                        "frequency_01": qubit.frequency,
                        "frequency_12": qubit.frequency + qubit.anharmonicity,
                        "maximum_frequency": qubit.frequency,
                        "sweetspot": qubit.sweetspot,
                        "asymmetry": qubit.asymmetry,
                    },
                    t1=(qubit.t1, None),
                    t2=(qubit.t2, None),
                )
                for target, qubit in model.qubits.items()
            }
        )
        return CalibrationPlatform(
            name=platform.name,
            parameters=platform.parameters,
            instruments=platform.instruments,
            qubits=platform.qubits,
            couplers=platform.couplers,
            resonator_type=platform.resonator_type,
            calibration=calibration,
        )
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def parameters(
    name: str,
    platform: CalibrationPlatform,
    targets: tuple[int, ...],
    *,
    points: int,
    nshots: int,
    full_window: bool = False,
    overrides: dict[str, object] | None = None,
) -> Parameters:
    """Small exploratory sweeps; full coherence windows are an explicit choice."""
    native = platform.natives.single_qubit[targets[0]].RX[0][1]
    amplitude_stop = min(1.0, 2.2 * native.amplitude)
    amplitude_step = amplitude_stop / points
    model = platform.config("hamiltonian")
    t1_window = 2 * max(model.qubits[q].t1 for q in targets) if full_window else 4000
    t2_window = 2 * max(model.qubits[q].t2 for q in targets) if full_window else 4000
    t1_step = max(2, int(t1_window / points))
    t2_step = max(2, int(t2_window / points))
    specs = {
        "rabi_amplitude": {
            "min_amp": amplitude_step / 2,
            "max_amp": amplitude_stop,
            "step_amp": amplitude_step,
            "rx90": False,
        },
        "rabi_length": {
            "pulse_duration_start": 4,
            "pulse_duration_end": 4 * (points + 1),
            "pulse_duration_step": 4,
            "pulse_amplitude": native.amplitude,
            "rx90": False,
            "interpolated_sweeper": False,
        },
        "ramsey": {"delay": (40, 40 + points * 250, 250), "detuning": 1e6},
        "t1": {
            "delay_before_readout_start": t1_step,
            "delay_before_readout_end": t1_step * (points + 1),
            "delay_before_readout_step": t1_step,
        },
        "t2": {
            "delay_between_pulses_start": t2_step,
            "delay_between_pulses_end": t2_step * (points + 1),
            "delay_between_pulses_step": t2_step,
        },
        "spin_echo": {
            "delay_between_pulses_start": max(native.duration, t2_step),
            "delay_between_pulses_end": max(native.duration, t2_step)
            + points * t2_step,
            "delay_between_pulses_step": t2_step,
        },
        "qubit_spectroscopy": {
            "freq_width": 30_000_000,
            "freq_step": 30_000_000 // points,
            "drive_duration": 100,
            "drive_amplitude": 0.015,
        },
        "flux_amplitude_frequency": {
            "amplitude_min": 0,
            "amplitude_max": 0.04,
            "amplitude_step": 0.04 / points,
            "duration": 100,
        },
    }
    inputs = specs[name] | {"nshots": nshots, "relaxation_time": 0}
    if overrides is not None:
        inputs.update(overrides)
    return PROTOCOLS[name].parameters_type.load(inputs)


@dataclass
class Session:
    name: str
    platform: CalibrationPlatform
    targets: tuple[int, ...]
    parameters: Parameters
    data: Data
    seconds: float
    fit_data: Data | None = None
    fit_seconds: float | None = None
    regularized_points: int = 0


def acquire(
    name: str,
    *,
    targets=(0, 1),
    coupled: bool | None = None,
    points: int = 17,
    nshots: int = 1024,
    full_window: bool = False,
    platform: CalibrationPlatform | None = None,
    overrides: dict[str, object] | None = None,
) -> Session:
    if name not in PROTOCOLS:
        raise ValueError(f"Unknown Qibocal protocol: {name!r}")
    targets = tuple(targets)
    if (
        not targets
        or len(set(targets)) != len(targets)
        or any(q not in (0, 1) for q in targets)
    ):
        raise ValueError("Select one or both distinct qutrit targets, 0 and 1")
    if points < 9 or nshots < 128:
        raise ValueError("Use at least nine sweep points and 128 shots")
    if platform is None:
        platform = calibration_platform(coupled=True if coupled is None else coupled)
    elif coupled is not None:
        raise ValueError("Choose coupling when creating the session platform")
    params = parameters(
        name,
        platform,
        targets,
        points=points,
        nshots=nshots,
        full_window=full_window,
        overrides=overrides,
    )
    if (
        isinstance(params.nshots, bool)
        or not isinstance(params.nshots, int)
        or params.nshots < 1
    ):
        raise ValueError("Protocol nshots must be a positive integer")
    platform.connect()
    try:
        data, seconds = PROTOCOLS[name].acquisition(params, platform, list(targets))
    finally:
        platform.disconnect()
    return Session(name, platform, targets, params, data, seconds)


def fit(session: Session, *, error_floor: bool = True) -> Results:
    """Use Qibocal's fit, optionally flooring finite-shot errors on a data copy."""
    data = deepcopy(session.data)
    regularized = 0
    if error_floor:
        for records in data.data.values():
            if "error" in records.dtype.names:
                floor = 1 / session.parameters.nshots
                regularized += np.count_nonzero(records["error"] < floor)
                records["error"] = np.maximum(records["error"], floor)
    fitted, seconds = PROTOCOLS[session.name].fit(data)
    missing = [target for target in session.targets if target not in fitted]
    if missing:
        raise RuntimeError(
            f"Qibocal {session.name} fit failed for {missing}; "
            "inspect the acquisition data and Qibocal logs"
        )
    session.fit_data = data
    session.fit_seconds = seconds
    session.regularized_points = regularized
    return fitted


def report(session: Session, fitted: Results):
    data = session.fit_data if session.fit_data is not None else session.data
    for target in session.targets:
        figures, table = PROTOCOLS[session.name].report(data, target=target, fit=fitted)
        yield target, figures, table


def apply_update(session: Session, fitted: Results) -> None:
    missing = [target for target in session.targets if target not in fitted]
    if missing:
        raise ValueError(f"Cannot update targets with missing fit results: {missing}")
    for target in session.targets:
        PROTOCOLS[session.name].update(fitted, session.platform, target=target)
