"""Batched rotating-frame dynamics with dynamiqs, on CPU, GPU, or TPU.

The model, frames, operators, segmentation, and validation are those of the QuTiP
engine. Experiments with the same structure (dimensions, dissipators, and active
time-dependent terms and their sample grids per segment) are solved together, one
dynamiqs call per segment. Each segment is rescaled to unit duration, so delays,
frequencies, amplitudes, and constant pulse durations may differ within a batch.

Time-independent segments are exponentiated exactly. Time-dependent segments use
an adaptive Runge-Kutta method, stepping exactly onto every sample boundary.
Optionally, frames follow the drives' detuning within each segment, entered and
left with diagonal phases, so that constant detuned drives are time-independent.
Batch, sample, and boundary counts are padded to powers of two, and coefficients
are passed as arrays, so compiled solvers are reused across segments and calls.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
from dynamiqs.time_qarray import (
    BatchedCallable,
    ModulatedTimeQArray,
    SummedTimeQArray,
)
from jax.tree_util import Partial
from numpy.typing import NDArray

from .engine import (
    _DEFAULT_SETTINGS,
    Drive,
    DynamiqsSettings,
    Flux,
    Model,
    Settings,
    _frame_shifts,
    _operators,
    checked_probabilities,
    classify,
    components,
    frame_frequencies,
    segments,
    validate,
)

Problem = tuple[Model, tuple[Drive, ...], tuple[Flux, ...], float]

_METHODS = {
    "dopri8": dq.method.Dopri8,
    "dopri5": dq.method.Dopri5,
    "tsit5": dq.method.Tsit5,
}
_DEFAULT_OPTIONS = DynamiqsSettings()


def _bucket(size: int) -> int:
    """Smallest power of two not below ``size``, bounding distinct compilations."""
    return 1 << max(size - 1, 0).bit_length()


@lru_cache(maxsize=16)
def _dense(model: Model):
    _, annihilation, numbers, drift, collapse, _ = _operators(model)
    return (
        [a.full() for a in annihilation],
        [n.full() for n in numbers],
        drift.full(),
        [c.full() for c in collapse],
    )


@dataclass(frozen=True)
class _Segment:
    start: float
    duration: float
    static: np.ndarray
    terms: tuple[tuple[Drive | Flux, float], ...]
    # Excitation numbers weighted by the frame shift, on the basis diagonal.
    frame: np.ndarray


def _segments(
    model: Model, drives, fluxes, duration, follow: bool = False
) -> list[_Segment]:
    """With ``follow``, frames shift by their drives' detuning where they agree."""
    annihilation, numbers, drift, _ = _dense(model)
    frames = frame_frequencies(model)
    roots = components(model)
    result = []
    for start, end, active in segments(drives, fluxes, duration):
        static = drift.copy()
        terms = []
        frame = np.zeros(len(drift))
        classified = classify(active, frames)
        shifts = _frame_shifts(classified, roots) if follow else None
        if shifts:
            for number, root in zip(numbers, roots, strict=True):
                if shift := shifts.get(root, 0.0):
                    static -= shift * number
                    frame += shift * np.diag(number).real
            classified = [replace(term, detuning=0.0) for term in classified]
        for term in classified:
            pulse = term.pulse
            if not term.static:
                terms.append((pulse, term.detuning))
            elif isinstance(pulse, Flux):
                static += pulse.samples[0] * numbers[pulse.target]
            else:
                a = annihilation[pulse.target]
                static += pulse.samples[0] * a.conj().T
                static += pulse.samples[0].conjugate() * a
        # A canonical order lets equally structured segments share a solver.
        terms.sort(key=lambda t: (isinstance(t[0], Flux), t[0].target))
        result.append(_Segment(start, end - start, static, tuple(terms), frame))
    return result


def _rotate(state, phases: np.ndarray, dissipative: bool):
    """Apply batched diagonal phases to kets, or on both sides of densities."""
    phases = jnp.asarray(phases)
    state = state * phases[:, :, None]
    return state * jnp.conj(phases)[:, None, :] if dissipative else state


def _grid(pulse: Drive | Flux, segment: _Segment) -> tuple | None:
    """Sample boundaries in the segment's rescaled time, if the pulse is shaped."""
    if len(pulse.samples) == 1:
        return None
    return (
        len(pulse.samples),
        round((pulse.start - segment.start) / segment.duration, 9),
        round(pulse.duration / segment.duration, 9),
    )


def _key(model: Model, problem_segments: list[_Segment]) -> tuple:
    """Levels and coherence times fix every operator except the Hamiltonian.

    Solvers step onto the sample boundaries of a whole batch, so only problems
    with the same rescaled sample grids share one: amplitude, phase, frequency,
    delay, and bias sweeps batch fully, while each duration forms its own batch.
    """
    return (
        tuple((q.levels, q.t1, q.t2) for q in model.qubits),
        tuple(
            tuple((isinstance(p, Flux), p.target, _grid(p, s)) for p, _ in s.terms)
            for s in problem_segments
        ),
    )


def structure(problem: Problem, options: DynamiqsSettings = _DEFAULT_OPTIONS) -> tuple:
    """The key of the batches ``problem`` can join: equal keys solve together."""
    validate(*problem, _DEFAULT_SETTINGS)
    model, drives, fluxes, duration = problem
    follow = options.frame == "drives"
    return _key(model, _segments(model, drives, fluxes, duration, follow))


def _samples_at(tau, origin, start, duration, count, samples, s):
    """Zero-order hold at rescaled time ``s``; arrays may carry a batch axis.

    Coefficients take the time last, as ``Partial`` binds the arrays first.
    """
    t = origin + s * tau
    index = jnp.clip(jnp.floor((t - start) * count / duration), 0, count - 1)
    value = jnp.take_along_axis(samples, index.astype(int)[..., None], axis=-1)
    return t, value[..., 0]


def _drive_coefficient(tau, origin, start, duration, count, samples, detuning, s):
    t, value = _samples_at(tau, origin, start, duration, count, samples, s)
    return tau * value * jnp.exp(-1j * detuning * t)


def _conjugate_coefficient(*args):
    return jnp.conj(_drive_coefficient(*args))


def _flux_coefficient(tau, origin, start, duration, count, samples, s):
    _, value = _samples_at(tau, origin, start, duration, count, samples, s)
    return tau * value + 0j


class _Coefficient(BatchedCallable):
    """A partially applied module-level function, whose arrays stay traced.

    ``dq.modulated``, ``BatchedCallable``, and broadcasting sums wrap callables in
    new closures or hashable shims, which JAX treats as static: the coefficient
    arrays would be baked into, and recompiled with, every solve.
    """

    def __init__(self, function, *arguments):
        self.f = Partial(function, *arguments)
        self.indices = list(jnp.indices(jax.eval_shape(self.f, 0.0).shape))


def _modulated(function, *arguments, operator, discontinuities):
    return ModulatedTimeQArray(
        _Coefficient(function, *arguments),
        dq.asqarray(operator),
        jnp.sort(discontinuities),
    )


def _boundaries(tau, origin, terms) -> np.ndarray:
    """Rescaled interior sample boundaries of every element, padded beyond 1."""
    points = [
        (
            pulse.start
            + np.arange(1, len(pulse.samples)) * pulse.duration / len(pulse.samples)
            - o
        )
        / t
        for t, o, element in zip(tau, origin, terms, strict=True)
        for pulse, _ in element
        if len(pulse.samples) > 1
    ]
    if not points:
        return np.zeros(0)
    interior = np.concatenate(points)
    interior = np.unique(interior[(interior > 0) & (interior < 1)].round(12))
    padded = np.full(_bucket(interior.size), 2.0)
    padded[: interior.size] = interior
    return padded


def _solve_group(
    model: Model,
    group: list[list[_Segment]],
    settings: Settings,
    options: DynamiqsSettings,
) -> NDArray:
    size = len(group)
    group = group + [group[0]] * (_bucket(size) - size)
    annihilation, numbers, _, collapse = _dense(model)
    dimension = annihilation[0].shape[0]
    dissipative = bool(collapse)
    state = np.zeros(
        (len(group), dimension, dimension if dissipative else 1), dtype=complex
    )
    state[:, 0, 0] = 1
    state = jnp.asarray(state)
    method = _METHODS[options.method](
        rtol=settings.rtol, atol=settings.atol, max_steps=settings.nsteps
    )
    for index in range(len(group[0])):
        column = [element[index] for element in group]
        tau = np.array([s.duration for s in column])
        origin = np.array([s.start for s in column])
        frame = np.stack([s.frame for s in column])
        shifted = bool(np.any(frame))
        if shifted:
            state = _rotate(state, np.exp(1j * origin[:, None] * frame), dissipative)
        static = dq.asqarray(
            jnp.asarray(tau[:, None, None] * np.stack([s.static for s in column]))
        )
        modulated = []
        terms = [s.terms for s in column]
        # Any one term can carry the boundaries of all the others.
        discontinuities = jnp.asarray(_boundaries(tau, origin, terms))
        for j in range(len(terms[0])):
            if j:
                discontinuities = jnp.zeros(0)
            pulses = [element[j][0] for element in terms]
            count = max(len(p.samples) for p in pulses)
            samples = np.stack(
                [
                    np.pad(p.samples, (0, _bucket(count) - len(p.samples)), "edge")
                    for p in pulses
                ]
            )
            arguments = tuple(
                jnp.asarray(a)
                for a in (
                    tau,
                    origin,
                    np.array([p.start for p in pulses]),
                    np.array([p.duration for p in pulses]),
                    np.array([len(p.samples) for p in pulses]),
                    samples,
                )
            )
            target = pulses[0].target
            if isinstance(pulses[0], Flux):
                modulated.append(
                    _modulated(
                        _flux_coefficient,
                        *arguments,
                        operator=numbers[target],
                        discontinuities=discontinuities,
                    )
                )
            else:
                arguments += (jnp.asarray([element[j][1] for element in terms]),)
                modulated += [
                    _modulated(
                        _drive_coefficient,
                        *arguments,
                        operator=annihilation[target].conj().T,
                        discontinuities=discontinuities,
                    ),
                    _modulated(
                        _conjugate_coefficient,
                        *arguments,
                        operator=annihilation[target],
                        discontinuities=discontinuities,
                    ),
                ]
        if modulated:
            # Every term has the same batch shape, so the sum needs no broadcast.
            hamiltonian = SummedTimeQArray([dq.constant(static), *modulated], False)
            solver = method
        else:
            hamiltonian, solver = static, dq.method.Expm()
        common = {
            "method": solver,
            "save_states": False,
            "cartesian_batching": False,
            "progress_meter": False,
        }
        if dissipative:
            scale = np.sqrt(tau)[:, None, None]
            jumps = [jnp.asarray(scale * c[None]) for c in collapse]
            result = dq.mesolve(hamiltonian, jumps, state, [0.0, 1.0], **common)
        else:
            result = dq.sesolve(hamiltonian, state, [0.0, 1.0], **common)
        state = result.final_state
        if dissipative and state.shape[-1] == 1:
            # Exponentiated Lindbladians return vectorized density matrices.
            state = dq.unvectorize(state)
        state = state.to_jax()
        if shifted:
            end = (origin + tau)[:, None]
            state = _rotate(state, np.exp(-1j * end * frame), dissipative)
        if dissipative:
            # Rounding, notably in single precision, breaks hermiticity, which
            # dynamiqs checks on the initial state of the next segment.
            state = (state + jnp.conj(jnp.swapaxes(state, -1, -2))) / 2
        if options.precision == "single":
            # The evolution preserves the trace, or the norm, which float32
            # rounding lets drift by about 1e-4 over long delays.
            if dissipative:
                trace = jnp.trace(state, axis1=-2, axis2=-1).real
                state = state / trace[:, None, None]
            else:
                state = state / jnp.linalg.norm(state, axis=(-2, -1))[:, None, None]
    final = np.asarray(state, dtype=np.complex128)[:size]
    if dissipative:
        return np.diagonal(final, axis1=-2, axis2=-1)
    return np.abs(final[..., 0]) ** 2


def evolve_batch(
    problems: Sequence[Problem],
    settings: Settings = _DEFAULT_SETTINGS,
    options: DynamiqsSettings = _DEFAULT_OPTIONS,
) -> list[NDArray[np.float64]]:
    """Terminal joint probabilities for each ``(model, drives, fluxes, duration)``.

    Equivalent to calling the QuTiP engine's ``evolve`` on every problem, up to
    solver accuracy.
    """
    if not isinstance(options, DynamiqsSettings):
        raise ValueError("Expected DynamiqsSettings")
    for problem in problems:
        validate(*problem, settings)
    dq.set_precision(options.precision)
    if options.device is not None:
        dq.set_device(options.device)
    floor = 1e-10 if options.precision == "double" else 1e3 * np.finfo("f4").eps
    groups: dict[tuple, list[int]] = {}
    split = []
    for index, (model, drives, fluxes, duration) in enumerate(problems):
        problem_segments = _segments(
            model, drives, fluxes, duration, options.frame == "drives"
        )
        split.append(problem_segments)
        # Different values are batched; only the structure has to agree.
        groups.setdefault(_key(model, problem_segments), []).append(index)
    results: list[NDArray[np.float64]] = [None] * len(problems)  # type: ignore[list-item]
    for indices in groups.values():
        # Operators come from the first member; batched values carry the rest.
        model = problems[indices[0]][0]
        probabilities = _solve_group(
            model, [split[i] for i in indices], settings, options
        )
        for i, p in zip(indices, probabilities, strict=True):
            results[i] = checked_probabilities(p, settings, floor)
    return results
