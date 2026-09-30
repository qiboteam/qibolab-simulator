"""Sparse rotating-frame dynamics, independent of Qibolab.

Frequencies are in Hz, times in ns, and sampled controls in radians/ns.
Each exchange-connected component uses a single, drive-independent frame.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite, pi, sqrt
from numbers import Integral, Real

import numpy as np
import qutip as qt
from numpy.typing import NDArray


def _real(name: str, value: float, *, positive: bool = False) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not isfinite(value)
        or (positive and value <= 0)
    ):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}real number")


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Qubit:
    """A truncated Duffing oscillator with optional Markovian decoherence."""

    frequency: float
    anharmonicity: float = 0.0
    levels: int = 2
    t1: float | None = None
    t2: float | None = None

    def __post_init__(self) -> None:
        _real("frequency", self.frequency, positive=True)
        _real("anharmonicity", self.anharmonicity)
        _integer("levels", self.levels, 2)
        for name, time in (("t1", self.t1), ("t2", self.t2)):
            if time is not None:
                _real(name, time, positive=True)
                _real(f"1/{name}", 1 / time, positive=True)
        if self.t1 is not None and self.t2 is not None and self.t2 / 2 > self.t1:
            raise ValueError("t2 must not exceed 2*t1")


@dataclass(frozen=True)
class Coupling:
    """Exchange coupling between distinct subsystem indices, in Hz."""

    first: int
    second: int
    strength: float

    def __post_init__(self) -> None:
        _integer("first", self.first)
        _integer("second", self.second)
        _real("strength", self.strength)
        if self.first == self.second:
            raise ValueError("A coupling must join distinct subsystems")


@dataclass(frozen=True)
class Model:
    qubits: tuple[Qubit, ...]
    couplings: tuple[Coupling, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.qubits, tuple)
            or not self.qubits
            or any(not isinstance(q, Qubit) for q in self.qubits)
        ):
            raise ValueError("qubits must be a nonempty tuple of Qubit objects")
        if not isinstance(self.couplings, tuple):
            raise ValueError("couplings must be a tuple")
        pairs = set()
        for coupling in self.couplings:
            if not isinstance(coupling, Coupling):
                raise ValueError("couplings must contain Coupling objects")
            if max(coupling.first, coupling.second) >= len(self.qubits):
                raise ValueError("Coupling target is outside the model")
            pair = tuple(sorted((coupling.first, coupling.second)))
            if pair in pairs:
                raise ValueError("Duplicate coupling")
            pairs.add(pair)


def _pulse_samples(pulse: "Drive | Flux", *, complex_samples: bool) -> np.ndarray:
    _integer("target", pulse.target)
    _real("start", pulse.start)
    _real("duration", pulse.duration)
    if pulse.start < 0 or pulse.duration < 0:
        raise ValueError("Pulse times must be nonnegative")
    end = pulse.start + pulse.duration
    if not isfinite(end) or (pulse.duration > 0 and end <= pulse.start):
        raise ValueError("Pulse end must be finite and resolve its duration")
    samples = np.asarray(pulse.samples)
    if (
        samples.ndim != 1
        or samples.dtype.kind not in "iufc"
        or (not complex_samples and np.iscomplexobj(samples))
        or (pulse.duration > 0 and samples.size == 0)
        or not np.all(np.isfinite(samples))
    ):
        raise ValueError("Samples must be finite, numeric, one-dimensional controls")
    samples = np.array(
        samples, dtype=np.complex128 if complex_samples else np.float64, copy=True
    )
    if not np.all(np.isfinite(samples)):
        raise ValueError("Samples overflow their control dtype")
    samples.flags.writeable = False
    return samples


@dataclass(frozen=True)
class Drive:
    """Creation-operator coefficients before the rotating-frame phase factor."""

    target: int
    start: float
    duration: float
    samples: NDArray[np.complex128]
    frequency: float

    def __post_init__(self) -> None:
        _real("frequency", self.frequency, positive=True)
        object.__setattr__(self, "samples", _pulse_samples(self, complex_samples=True))


@dataclass(frozen=True)
class Flux:
    """Angular frequency shifts from the model's DC-biased qubit frequency."""

    target: int
    start: float
    duration: float
    samples: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", _pulse_samples(self, complex_samples=False))


@dataclass(frozen=True)
class Settings:
    rtol: float = 1e-7
    atol: float = 1e-9
    nsteps: int = 100000

    def __post_init__(self) -> None:
        _real("rtol", self.rtol, positive=True)
        _real("atol", self.atol, positive=True)
        _integer("nsteps", self.nsteps, 1)


def frame_frequencies(model: Model) -> tuple[float, ...]:
    """Use the lowest-index qubit's frequency for each connected component."""
    if not isinstance(model, Model):
        raise ValueError("model must be a Model")
    roots = list(range(len(model.qubits)))

    def root(index: int) -> int:
        while roots[index] != index:
            index = roots[index]
        return index

    for coupling in model.couplings:
        first, second = root(coupling.first), root(coupling.second)
        roots[max(first, second)] = min(first, second)
    return tuple(model.qubits[root(i)].frequency for i in range(len(roots)))


@lru_cache(maxsize=16)
def _operators(model: Model):
    """Cache only sparse, drive-independent operators and the initial ket."""
    frames = frame_frequencies(model)
    identities = [qt.qeye(q.levels, dtype="csr") for q in model.qubits]
    annihilation, numbers = [], []
    for i, qubit in enumerate(model.qubits):
        factors = identities.copy()
        factors[i] = qt.destroy(qubit.levels, dtype="csr")
        a = qt.tensor(factors)
        annihilation.append(a)
        numbers.append(a.dag() * a)
    identity = qt.tensor(identities)
    hamiltonian = 0 * identity
    collapse = []
    for qubit, frame, a, n in zip(
        model.qubits, frames, annihilation, numbers, strict=True
    ):
        hamiltonian += (2 * pi / 1e9) * (qubit.frequency - frame) * n
        hamiltonian += (pi / 1e9) * qubit.anharmonicity * n * (n - identity)
        relaxation = 0.0 if qubit.t1 is None else 1 / qubit.t1
        if relaxation:
            collapse.append(sqrt(relaxation) * a)
        if qubit.t2 is not None:
            dephasing = max(0.0, 1 / qubit.t2 - relaxation / 2)
            if dephasing:
                collapse.append((sqrt(2) * sqrt(dephasing)) * n)
    for coupling in model.couplings:
        exchange = annihilation[coupling.first].dag() * annihilation[coupling.second]
        hamiltonian += (2 * pi / 1e9) * coupling.strength * (exchange + exchange.dag())
    ground = qt.tensor([qt.basis(q.levels, 0) for q in model.qubits])
    return frames, annihilation, numbers, hamiltonian, collapse, ground


def _coefficient(pulse: Drive | Flux, origin: float, detuning: float = 0.0):
    """Zero-order hold; callers restrict integration to this pulse's support."""
    samples = pulse.samples
    constant = np.all(samples == samples[0])
    scale = len(samples) / pulse.duration

    def coefficient(t):
        index = (
            0
            if constant
            else min(
                max(int((t + (origin - pulse.start)) * scale), 0), len(samples) - 1
            )
        )
        value = samples[index]
        return value * np.exp(-1j * detuning * (t + origin)) if detuning else value

    return coefficient


_DEFAULT_SETTINGS = Settings()


def evolve(
    model: Model,
    drives: tuple[Drive, ...],
    fluxes: tuple[Flux, ...],
    duration: float,
    settings: Settings = _DEFAULT_SETTINGS,
) -> NDArray[np.float64]:
    """Return terminal joint basis probabilities, with the last subsystem fastest.

    All pulse boundaries restart integration. Changing sample sequences additionally
    constrain the adaptive step, but constant envelopes and idle intervals do not.
    Closed, diagonal autonomous segments apply exact phases without integration.
    Only the terminal state is retained; operators never become dense arrays.
    """
    if not isinstance(model, Model) or not isinstance(settings, Settings):
        raise ValueError("Expected a Model and Settings")
    _real("duration", duration)
    if duration < 0:
        raise ValueError("duration must be nonnegative")
    for pulses, kind in ((drives, Drive), (fluxes, Flux)):
        if not isinstance(pulses, tuple) or any(
            not isinstance(pulse, kind) for pulse in pulses
        ):
            raise ValueError(f"Expected a tuple of {kind.__name__} objects")
        for pulse in pulses:
            if pulse.target >= len(model.qubits):
                raise ValueError("Pulse target is outside the model")
            if pulse.start + pulse.duration > duration + 1e-9:
                raise ValueError("Pulse extends beyond evolution duration")
    frames, annihilation, numbers, drift, collapse, state = _operators(model)
    pulses = tuple(p for p in (*drives, *fluxes) if p.duration > 0)
    boundaries = sorted(
        {0.0, duration}
        | {min(duration, t) for p in pulses for t in (p.start, p.start + p.duration)}
    )
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        active = [p for p in pulses if p.start < end and p.start + p.duration > start]
        static = drift
        terms = []
        max_step = 0.0
        for pulse in active:
            changing = np.any(pulse.samples != pulse.samples[0])
            if changing:
                spacing = pulse.duration / len(pulse.samples) / 2
                max_step = min(max_step, spacing) if max_step else spacing
            if isinstance(pulse, Flux):
                if changing:
                    terms.append([numbers[pulse.target], _coefficient(pulse, start)])
                else:
                    static = static + pulse.samples[0] * numbers[pulse.target]
            else:
                a = annihilation[pulse.target]
                detuning = (2 * pi / 1e9) * (pulse.frequency - frames[pulse.target])
                if not changing and detuning == 0:
                    static = static + pulse.samples[0] * a.dag()
                    static = static + pulse.samples[0].conjugate() * a
                else:
                    coefficient = _coefficient(pulse, start, detuning)
                    terms.extend(
                        [
                            [a.dag(), coefficient],
                            [a, lambda t, c=coefficient: np.conjugate(c(t))],
                        ]
                    )
        if not collapse and not terms and qt.core.data.isdiag(static.data):
            phases = np.exp(-1j * (end - start) * static.diag())
            state = qt.Qobj(phases[:, None] * state.full(), dims=state.dims, copy=False)
            continue
        hamiltonian = [static, *terms] if terms else static
        options = {
            "rtol": settings.rtol,
            "atol": settings.atol,
            "nsteps": settings.nsteps,
            "max_step": max_step,
            "method": "dop853",
            "store_states": False,
            "store_final_state": True,
            "normalize_output": False,
            "progress_bar": "",
        }
        if collapse:
            result = qt.mesolve(
                hamiltonian, state, [0, end - start], c_ops=collapse, options=options
            )
        else:
            result = qt.sesolve(hamiltonian, state, [0, end - start], options=options)
        state = result.final_state
    if state.isket:
        probabilities = np.abs(state.full().ravel()) ** 2
    elif isinstance(state.data, qt.core.data.Dense):
        # Qobj.diag() can discard NaNs while converting its dense input to CSR.
        probabilities = state.data.as_ndarray().diagonal()
    else:
        probabilities = state.data.as_scipy().diagonal()
    tolerance = max(1e-10, min(1e-5, 20 * (settings.atol + settings.rtol)))
    if (
        not np.all(np.isfinite(probabilities))
        or np.max(np.abs(np.imag(probabilities))) > tolerance
        or np.min(np.real(probabilities)) < -tolerance
        or abs(np.sum(probabilities) - 1) > tolerance
    ):
        raise RuntimeError("Solver produced invalid or unnormalized probabilities")
    probabilities = np.clip(np.real(probabilities), 0, 1)
    return probabilities / probabilities.sum()
