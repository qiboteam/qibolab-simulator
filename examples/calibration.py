"""Bind Qibocal's built-in routines to its direct protocol executor."""

import signal
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from plotly.graph_objects import Figure
from qibocal import Executor, protocols
from qibocal.auto.operation import BoundProtocol, Data, Parameters, Protocol, Results
from qibocal.calibration import CalibrationPlatform
from qibocal.calibration.calibration import Calibration, QubitCalibration

from examples.platforms.split_transmons import create_platform

Report = tuple[int, list[Figure], str]

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
class UpdatePlatform:
    """Bridge Executor.update(results) to a routine's per-target update."""

    platform: CalibrationPlatform
    routine: Protocol
    targets: tuple[int, ...]

    def update(self, results: Results) -> None:
        require_results(results, self.targets)
        if self.routine.update is None:
            raise ValueError("Protocol does not support updating")
        for target in self.targets:
            self.routine.update(results, self.platform, target=target)


def require_results(results: Results, targets: tuple[int, ...]) -> None:
    missing = [target for target in targets if target not in results]
    if missing:
        raise ValueError(
            f"Qibocal fit has no results for {missing}; "
            "inspect the acquisition data and Qibocal logs"
        )


def bind(
    name: str,
    *,
    targets=(0, 1),
    coupled: bool | None = None,
    points: int = 17,
    nshots: int = 1024,
    full_window: bool = False,
    platform: CalibrationPlatform | None = None,
    overrides: dict[str, object] | None = None,
    error_floor: bool = True,
    reports: list[Report] | None = None,
) -> tuple[Executor, BoundProtocol]:
    """Adapt legacy routine signatures without executing any protocol phases.

    The PR has not migrated built-in routines yet. Acquisition captures the
    hardware and targets; report outputs are collected in the supplied list.
    Updates remain explicit through Executor.update and never write a runcard.
    """
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
    routine = PROTOCOLS[name]

    def acquisition(parameters: Parameters) -> Data:
        platform.connect()
        try:
            return routine.acquisition(parameters, platform, list(targets))
        finally:
            platform.disconnect()

    def fit(data: Data, floor_errors: bool | None) -> Results:
        if routine.fit is None:
            raise ValueError("Protocol does not support fitting")
        fit_data = deepcopy(data)
        if floor_errors:
            for records in fit_data.data.values():
                if records.dtype.names is not None and "error" in records.dtype.names:
                    records["error"] = np.maximum(records["error"], 1 / params.nshots)
        fitted = routine.fit(fit_data)
        require_results(fitted, targets)
        return fitted

    def report(data: Data, fitted: Results, outputs: list[Report] | None) -> None:
        if routine.report is None:
            raise ValueError("Protocol does not support reporting")
        if outputs is None:
            raise ValueError("Supply a report output list when binding the protocol")
        for target in targets:
            figures, table = routine.report(data, target=target, fit=fitted)
            outputs.append((target, figures, table))

    def update(fitted: Results, platform: CalibrationPlatform) -> None:
        UpdatePlatform(platform, routine, targets).update(fitted)

    protocol = Protocol(
        acquisition=acquisition,
        fit=fit if routine.fit is not None else None,
        report=report if reports is not None else None,
        update=update if routine.update is not None else None,
    )
    bound = protocol(pars=params, fit=error_floor, report=reports)
    return Executor(UpdatePlatform(platform, routine, targets)), bound
