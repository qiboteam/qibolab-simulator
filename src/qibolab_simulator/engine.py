"""Sparse rotating-frame dynamics, independent of Qibolab.

Frequencies are in Hz, times in ns, and sampled controls in radians/ns.
Each exchange-connected component uses a single, drive-independent frame.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite, pi, sqrt
from numbers import Integral, Real
from typing import Literal

import numpy as np
import qutip as qt
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply


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
    """Solver accuracy, shared by the QuTiP and dynamiqs backends.

    ``propagation="exact"`` makes the QuTiP backend exponentiate every interval
    between sample edges, instead of integrating adaptively; tolerances then only
    apply to segments falling back to the adaptive solver. dynamiqs ignores it.
    """

    rtol: float = 1e-7
    atol: float = 1e-9
    nsteps: int = 100000
    propagation: Literal["adaptive", "exact"] = "adaptive"

    def __post_init__(self) -> None:
        _real("rtol", self.rtol, positive=True)
        _real("atol", self.atol, positive=True)
        _integer("nsteps", self.nsteps, 1)
        if self.propagation not in ("adaptive", "exact"):
            raise ValueError("propagation must be 'adaptive' or 'exact'")


@dataclass(frozen=True)
class DynamiqsSettings:
    """Options of the dynamiqs backend only; the QuTiP backend ignores them.

    ``precision`` and ``device`` set JAX's process-wide defaults when solving.
    ``device=None`` keeps JAX's default, which is a GPU when one is available.
    """

    method: Literal["dopri8", "dopri5", "tsit5"] = "dopri8"
    precision: Literal["double", "single"] = "double"
    device: Literal["cpu", "gpu", "tpu"] | None = None

    def __post_init__(self) -> None:
        for name, allowed in (
            ("method", ("dopri8", "dopri5", "tsit5")),
            ("precision", ("double", "single")),
            ("device", (None, "cpu", "gpu", "tpu")),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")


def components(model: Model) -> tuple[int, ...]:
    """The lowest qubit index of each qubit's exchange-connected component."""
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
    return tuple(root(i) for i in range(len(roots)))


def frame_frequencies(model: Model) -> tuple[float, ...]:
    """Use the lowest-index qubit's frequency for each connected component."""
    return tuple(model.qubits[root].frequency for root in components(model))


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


def validate(
    model: Model,
    drives: tuple[Drive, ...],
    fluxes: tuple[Flux, ...],
    duration: float,
    settings: Settings,
) -> None:
    """Check one evolution request; shared by all solver backends."""
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


def segments(
    drives: tuple[Drive, ...], fluxes: tuple[Flux, ...], duration: float
) -> list[tuple[float, float, list[Drive | Flux]]]:
    """Split ``[0, duration]`` at every pulse boundary, listing the active pulses."""
    pulses = tuple(p for p in (*drives, *fluxes) if p.duration > 0)
    boundaries = sorted(
        {0.0, duration}
        | {min(duration, t) for p in pulses for t in (p.start, p.start + p.duration)}
    )
    return [
        (
            start,
            end,
            [p for p in pulses if p.start < end and p.start + p.duration > start],
        )
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


@dataclass(frozen=True)
class Term:
    """An active pulse; static terms join the time-independent Hamiltonian."""

    pulse: Drive | Flux
    changing: bool
    detuning: float = 0.0

    @property
    def static(self) -> bool:
        return not self.changing and self.detuning == 0


def classify(active: list[Drive | Flux], frames: tuple[float, ...]) -> list[Term]:
    """Detunings are angular frequencies from each target's rotating frame."""
    return [
        Term(
            pulse,
            bool(np.any(pulse.samples != pulse.samples[0])),
            (2 * pi / 1e9) * (pulse.frequency - frames[pulse.target])
            if isinstance(pulse, Drive)
            else 0.0,
        )
        for pulse in active
    ]


# Generators up to this dimension, D for kets and D² for density matrices, are
# exponentiated densely and cached; larger ones only apply their action.
_DENSE_LIMIT = 256
_CACHE_BYTES = 1 << 27


@lru_cache(maxsize=16)
def _generators(model: Model, dense: bool):
    """Generators of the drift and of every control operator.

    Kets evolve with ``-iH``; column-stacked density matrices with the Lindbladian,
    whose control terms are commutators. Number generators are diagonal.
    """
    _, annihilation, numbers, drift, collapse, _ = _operators(model)

    def generator(operator, dissipators=()):
        if collapse:
            result = -1j * (qt.spre(operator) - qt.spost(operator))
            if dissipators:
                result += qt.liouvillian(None, list(dissipators))
        else:
            result = -1j * operator
        matrix = result.to("csr").data.as_scipy().tocsr()
        return matrix.toarray() if dense else matrix

    return (
        generator(drift, collapse),
        tuple(generator(a.dag()) for a in annihilation),
        tuple(generator(a) for a in annihilation),
        tuple(generator(n) for n in numbers),
    )


def _frame_shifts(terms: list[Term], roots: tuple[int, ...]) -> dict[int, float] | None:
    """The detuning shared by all drives of each component, or ``None``."""
    shifts = {}
    for term in terms:
        if isinstance(term.pulse, Drive):
            root = roots[term.pulse.target]
            if shifts.setdefault(root, term.detuning) != term.detuning:
                return None
    return shifts


def _edges(terms: list[Term], start: float, end: float, tolerance: float = 1e-9):
    """Split ``[start, end]`` at the sample edges of every changing pulse."""
    interior = np.unique(
        np.concatenate(
            [
                p.start + np.arange(1, len(p.samples)) * (p.duration / len(p.samples))
                for p in (term.pulse for term in terms if term.changing)
            ]
            or [np.zeros(0)]
        )
    )
    interior = interior[(interior > start + tolerance) & (interior < end - tolerance)]
    interior = interior[np.diff(interior, prepend=-np.inf) > tolerance]
    return [start, *interior, end]


def _sample(pulse: Drive | Flux, time: float):
    index = int((time - pulse.start) * len(pulse.samples) / pulse.duration)
    return pulse.samples[min(max(index, 0), len(pulse.samples) - 1)]


def _propagate(
    model: Model,
    terms: list[Term],
    shifts: dict[int, float],
    start: float,
    end: float,
    state: qt.Qobj,
    cache: dict,
) -> qt.Qobj:
    """Exponentiate each zero-order-hold interval, in a frame following the drives.

    Shifting each component's frame by its drives' detuning ``d`` removes their
    ``exp(-idt)`` phases, adding ``-d`` times the component's excitation number to
    the Hamiltonian. Exchange conserves that number, and dissipators are invariant
    under the frame's phases, so the generator is constant between sample edges.
    """
    collapse = _operators(model)[4]
    if collapse and state.isket:
        state = state.proj()
    vector = state.full().ravel(order="F")
    dense = vector.size <= _DENSE_LIMIT
    drift, raising, lowering, numbers = _generators(model, dense)
    detunings = [shifts.get(root, 0.0) for root in components(model)]
    # The diagonal of -i times the components' excitation numbers, weighted.
    frame = sum(
        (d * n.diagonal() for d, n in zip(detunings, numbers, strict=True) if d),
        np.zeros(vector.size),
    )
    vector = np.exp(-start * frame) * vector
    capacity = _CACHE_BYTES // (16 * vector.size**2)
    edges = _edges(terms, start, end)
    for first, second in zip(edges[:-1], edges[1:], strict=True):
        coefficients = {("flux", i): -d for i, d in enumerate(detunings) if d}
        for term in terms:
            pulse = term.pulse
            value = _sample(pulse, (first + second) / 2)
            key = ("flux" if isinstance(pulse, Flux) else "drive", pulse.target)
            coefficients[key] = coefficients.get(key, 0) + value
        key = (round(second - first, 9), tuple(sorted(coefficients.items())))
        propagator = cache.get(key) if dense else None
        if propagator is None:
            generator = drift
            for (kind, target), value in coefficients.items():
                if kind == "flux":
                    generator = generator + value * numbers[target]
                else:
                    generator = generator + value * raising[target]
                    generator = generator + np.conj(value) * lowering[target]
            if not dense:
                vector = expm_multiply((second - first) * generator, vector)
                continue
            propagator = expm((second - first) * generator)
            if len(cache) < capacity:
                cache[key] = propagator
        vector = propagator @ vector
    vector = np.exp(end * frame) * vector
    if collapse:
        dimension = state.shape[0]
        vector = vector.reshape((dimension, dimension), order="F")
    else:
        vector = vector[:, None]
    return qt.Qobj(vector, dims=state.dims, copy=False)


def checked_probabilities(
    probabilities: np.ndarray, settings: Settings, floor: float = 1e-10
) -> NDArray[np.float64]:
    """Reject invalid solver output, then clip and renormalize it."""
    tolerance = max(floor, min(1e-5, 20 * (settings.atol + settings.rtol)))
    if (
        not np.all(np.isfinite(probabilities))
        or np.max(np.abs(np.imag(probabilities))) > tolerance
        or np.min(np.real(probabilities)) < -tolerance
        or abs(np.sum(probabilities) - 1) > tolerance
    ):
        raise RuntimeError("Solver produced invalid or unnormalized probabilities")
    probabilities = np.clip(np.real(probabilities), 0, 1)
    return probabilities / probabilities.sum()


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
    Exact propagation exponentiates the other segments, unless a component has
    drives with different detunings. Only the terminal state is retained; adaptive
    and large exact propagation never make operators dense arrays.
    """
    validate(model, drives, fluxes, duration, settings)
    frames, annihilation, numbers, drift, collapse, state = _operators(model)
    roots = components(model)
    cache = {}
    for start, end, active in segments(drives, fluxes, duration):
        static = drift
        terms = []
        max_step = 0.0
        classified = classify(active, frames)
        for term in classified:
            pulse = term.pulse
            if term.changing:
                spacing = pulse.duration / len(pulse.samples) / 2
                max_step = min(max_step, spacing) if max_step else spacing
            if isinstance(pulse, Flux):
                if term.static:
                    static = static + pulse.samples[0] * numbers[pulse.target]
                else:
                    terms.append([numbers[pulse.target], _coefficient(pulse, start)])
            else:
                a = annihilation[pulse.target]
                if term.static:
                    static = static + pulse.samples[0] * a.dag()
                    static = static + pulse.samples[0].conjugate() * a
                else:
                    coefficient = _coefficient(pulse, start, term.detuning)
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
        if settings.propagation == "exact":
            shifts = _frame_shifts(classified, roots)
            if shifts is not None:
                state = _propagate(model, classified, shifts, start, end, state, cache)
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
    return checked_probabilities(probabilities, settings)
