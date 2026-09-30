"""Compile a Qibolab sequence into a small, serializable solver job."""

from dataclasses import dataclass

import numpy as np
from qibolab import (
    Acquisition,
    Config,
    Delay,
    Pulse,
    PulseSequence,
    Readout,
    Rectangular,
    VirtualZ,
)
from qibolab._core.pulses import PulseId

from .config import DriveConfig, FluxConfig, HamiltonianConfig, ReadoutConfig
from .engine import Coupling, Drive, Flux, Model, Qubit


@dataclass(frozen=True)
class Experiment:
    model: Model
    drives: tuple[Drive, ...]
    fluxes: tuple[Flux, ...]
    duration: float
    acquisitions: tuple[tuple[PulseId, int], ...]


def compile_sequence(
    sequence: PulseSequence, configs: dict[str, Config], sampling_rate: float
) -> Experiment:
    """Resolve timing after sweeping, including Align and VirtualZ instructions."""
    hconfig = configs.get("hamiltonian")
    if not isinstance(hconfig, HamiltonianConfig):
        raise ValueError("'hamiltonian' must contain a simulator HamiltonianConfig")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    indices = {key: i for i, key in enumerate(hconfig.qubits)}
    biases = {}
    for channel, config in configs.items():
        if isinstance(config, (DriveConfig, FluxConfig, ReadoutConfig)):
            if config.target not in indices:
                raise ValueError(f"Unknown target {config.target!r} for {channel!r}")
        if isinstance(config, FluxConfig):
            if config.target in biases:
                raise ValueError("Only one flux channel per subsystem is supported")
            biases[config.target] = config.offset * config.voltage_to_flux
    qubits = tuple(
        Qubit(
            frequency=float(q.biased_frequency(biases[key]))
            if key in biases
            else q.frequency,
            anharmonicity=q.anharmonicity,
            levels=q.levels,
            t1=q.t1,
            t2=q.t2,
        )
        for key, q in hconfig.qubits.items()
    )
    model = Model(
        qubits,
        tuple(
            Coupling(indices[p.qubits[0]], indices[p.qubits[1]], p.coupling)
            for p in hconfig.couplings
        ),
    )
    drives, fluxes, acquisitions, measurement_times = [], [], [], []
    times, phases = {}, {}
    control_end = 0.0
    for channel, pulse in sequence.align_to_delays():
        if channel not in configs:
            raise ValueError(f"No configuration for channel {channel!r}")
        config = configs[channel]
        if not isinstance(config, (DriveConfig, FluxConfig, ReadoutConfig)):
            raise NotImplementedError(f"Unsupported channel configuration: {channel!r}")
        time = times.get(channel, 0.0)
        duration = pulse.duration
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("Instruction durations must be finite and nonnegative")
        if isinstance(pulse, Delay):
            pass
        elif isinstance(pulse, VirtualZ):
            if not isinstance(config, DriveConfig):
                raise ValueError("VirtualZ requires a drive channel")
            if not np.isfinite(pulse.phase):
                raise ValueError("VirtualZ phase must be finite")
            phases[channel] = phases.get(channel, 0.0) + pulse.phase
        elif isinstance(pulse, (Acquisition, Readout)):
            if not isinstance(config, ReadoutConfig):
                raise ValueError("Acquisition requires a simulator readout channel")
            if isinstance(pulse, Readout) and pulse.time_of_flight != 0:
                raise NotImplementedError("Readout time_of_flight is not modeled")
            if config.delay != 0 or config.smearing != 0:
                raise NotImplementedError("Readout delay and smearing are not modeled")
            acquisitions.append((pulse.id, indices[config.target]))
            measurement_times.append(time)
        elif isinstance(pulse, Pulse):
            if not isinstance(config, (DriveConfig, FluxConfig)):
                raise ValueError("A control pulse requires a drive or flux channel")
            if getattr(pulse, "chirp", None) is not None:
                raise NotImplementedError("Chirped pulses are not supported")
            if not np.isfinite(pulse.amplitude) or abs(pulse.amplitude) > 1:
                raise ValueError("Pulse amplitude must be finite and in [-1, 1]")
            if not np.isfinite(pulse.relative_phase):
                raise ValueError("Pulse phase must be finite")
            count = round(duration * sampling_rate)
            if (duration > 0 and count == 0) or not np.isclose(
                duration * sampling_rate, count, rtol=0, atol=1e-8
            ):
                raise ValueError("Pulse duration must lie on the sampling grid")
            if duration:
                # Use the validated integer count rather than truncating duration*rate.
                sample_count = 1 if isinstance(pulse.envelope, Rectangular) else count
                envelopes = pulse.amplitude * np.array(
                    [pulse.envelope.i(sample_count), pulse.envelope.q(sample_count)]
                )
                if envelopes.shape != (2, sample_count) or not np.all(
                    np.isfinite(envelopes)
                ):
                    raise ValueError(
                        "Pulse samples must be finite and match the sample count"
                    )
                i, q = envelopes
                target = indices[config.target]
                if isinstance(config, DriveConfig):
                    phase = pulse.relative_phase + phases.get(channel, 0.0)
                    samples = (
                        -1j
                        * np.pi
                        * config.rabi_frequency
                        / 1e9
                        * (i + 1j * q)
                        * np.exp(-1j * phase)
                    )
                    drives.append(
                        Drive(target, time, duration, samples, config.frequency)
                    )
                else:
                    if np.any(q != 0) or pulse.relative_phase != 0:
                        raise NotImplementedError(
                            "Flux pulses must have real envelopes"
                        )
                    qubit = hconfig.qubits[config.target]
                    frequency = qubit.biased_frequency(
                        config.voltage_to_flux * (config.offset + i)
                    )
                    if np.any(frequency <= 0):
                        raise ValueError(
                            "Flux pulse leaves the positive-frequency model"
                        )
                    samples = 2 * np.pi * (frequency - qubits[target].frequency) / 1e9
                    fluxes.append(Flux(target, time, duration, samples))
                control_end = max(control_end, time + duration)
        else:
            raise NotImplementedError(
                f"Unsupported instruction: {type(pulse).__name__}"
            )
        times[channel] = time + duration
    if not acquisitions:
        raise ValueError("Each sequence must contain a terminal acquisition")
    if len({identifier for identifier, _ in acquisitions}) != len(acquisitions):
        raise ValueError("Acquisition identifiers must be unique")
    terminal = measurement_times[0]
    if any(abs(t - terminal) > 1e-9 for t in measurement_times):
        raise NotImplementedError(
            "Only simultaneous terminal acquisitions are supported"
        )
    if control_end > terminal + 1e-9:
        raise NotImplementedError("Control pulses cannot extend beyond measurement")
    return Experiment(
        model, tuple(drives), tuple(fluxes), terminal, tuple(acquisitions)
    )
