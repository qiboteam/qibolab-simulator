"""Analytic physics checks in ns and Hz, without Qibolab pulse compilation."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import qutip as qt

from qibolab_simulator.engine import (
    Coupling,
    Drive,
    Flux,
    Model,
    Qubit,
    Settings,
    _operators,
    evolve,
    frame_frequencies,
)

FREQUENCY = 5e9
STRICT = Settings(rtol=1e-9, atol=1e-11)


def rotation(angle, *, start=0, duration=20, phase=0, frequency=FREQUENCY, target=0):
    return Drive(
        target,
        start,
        duration,
        np.array([-1j * angle / (2 * duration) * np.exp(-1j * phase)]),
        frequency,
    )


@pytest.mark.parametrize("angle", [0, np.pi / 2, np.pi, 2 * np.pi])
def test_resonant_rabi(angle):
    result = evolve(Model((Qubit(FREQUENCY),)), (rotation(angle),), (), 20, STRICT)
    np.testing.assert_allclose(
        result, [np.cos(angle / 2) ** 2, np.sin(angle / 2) ** 2], atol=1e-8
    )


@pytest.mark.parametrize("detuning", [-80e6, -5e6, 0, 5e6, 80e6])
def test_detuned_rabi(detuning):
    duration, rabi = 63.0, 13e6
    omega, delta = 2 * np.pi * rabi / 1e9, 2 * np.pi * detuning / 1e9
    pulse = rotation(
        omega * duration, duration=duration, frequency=FREQUENCY + detuning
    )
    result = evolve(Model((Qubit(FREQUENCY),)), (pulse,), (), duration, STRICT)
    rate = np.hypot(omega, delta)
    expected = omega**2 / rate**2 * np.sin(rate * duration / 2) ** 2
    assert result[1] == pytest.approx(expected, abs=1e-8)


def test_lab_carrier_invariance():
    model = Model((Qubit(FREQUENCY), Qubit(FREQUENCY + 30e6)), (Coupling(0, 1, 8e6),))
    pulses = (
        rotation(2.0, frequency=FREQUENCY + 11e6, duration=29),
        rotation(1.3, frequency=FREQUENCY + 19e6, duration=19, start=13, target=1),
    )
    offset = 137e9
    shifted = Model(
        tuple(replace(q, frequency=q.frequency + offset) for q in model.qubits),
        model.couplings,
    )
    np.testing.assert_allclose(
        evolve(model, pulses, (), 45, STRICT),
        evolve(
            shifted,
            tuple(replace(p, frequency=p.frequency + offset) for p in pulses),
            (),
            45,
            STRICT,
        ),
        atol=1e-10,
    )


@pytest.mark.parametrize("phase", [0, np.pi / 2, np.pi, -np.pi / 2])
def test_relative_phase_and_quadrature(phase):
    first = rotation(np.pi / 2)
    second = rotation(np.pi / 2, start=20, phase=phase)
    model = Model((Qubit(FREQUENCY),))
    expected = (1 + np.cos(phase)) / 2
    assert evolve(model, (first, second), (), 40, STRICT)[1] == pytest.approx(
        expected, abs=1e-8
    )
    # The Q quadrature enters as -i*(I+iQ), not as a real envelope magnitude.
    quadrature = replace(
        second,
        samples=-1j
        * (np.array([np.cos(phase)]) - 1j * np.array([np.sin(phase)]))
        * np.pi
        / 80,
    )
    np.testing.assert_allclose(
        evolve(model, (first, quadrature), (), 40, STRICT),
        evolve(model, (first, second), (), 40, STRICT),
        atol=1e-9,
    )


def test_shaped_quadrature_zero_order_hold():
    samples = np.array([0, -0.06j, 0.04 - 0.08j, -0.04 - 0.08j, -0.06j, 0])
    pulse = Drive(0, 7, 30, samples, FREQUENCY)
    state = qt.basis(2, 0)
    a = qt.destroy(2)
    for sample in samples:
        state = (-1j * 5 * (sample * a.dag() + sample.conjugate() * a)).expm() * state
    expected = np.abs(state.full().ravel()) ** 2
    result = evolve(Model((Qubit(FREQUENCY),)), (pulse,), (), 41, STRICT)
    np.testing.assert_allclose(result, expected, atol=2e-7)


def test_t1_decay():
    model = Model((Qubit(FREQUENCY, t1=130),))
    pulse = rotation(np.pi, duration=3)
    prepared = evolve(model, (pulse,), (), 3, STRICT)[1]
    result = evolve(model, (pulse,), (), 83, STRICT)[1]
    assert result == pytest.approx(prepared * np.exp(-80 / 130), abs=1e-8)


@pytest.mark.parametrize("t1", [None, 200.0])
def test_t2_ramsey(t1):
    t2, delay, width = 120, 71, 1e-3
    pulses = (
        rotation(np.pi / 2, duration=width),
        rotation(np.pi / 2, duration=width, start=width + delay),
    )
    result = evolve(
        Model((Qubit(FREQUENCY, t1=t1, t2=t2),)), pulses, (), delay + 2 * width, STRICT
    )
    assert result[1] == pytest.approx((1 + np.exp(-delay / t2)) / 2, abs=2e-5)


def test_absolute_time_phase_between_detuned_pulses():
    detuning, delay, width = 7e6, 49, 1e-3
    pulses = (
        rotation(np.pi / 2, frequency=FREQUENCY + detuning, duration=width),
        rotation(
            np.pi / 2,
            frequency=FREQUENCY + detuning,
            duration=width,
            start=width + delay,
        ),
    )
    result = evolve(Model((Qubit(FREQUENCY),)), pulses, (), delay + 2 * width, STRICT)
    expected = (1 + np.cos(2 * np.pi * detuning * (delay + width) / 1e9)) / 2
    assert result[1] == pytest.approx(expected, abs=2e-5)


def test_flux_ramsey():
    width, delay, shift = 1.0, 40.0, 0.037
    pulses = (
        rotation(np.pi / 2, duration=width),
        rotation(np.pi / 2, start=width + delay, duration=width),
    )
    flux = Flux(0, width, delay, np.full(2000, shift))
    result = evolve(
        Model((Qubit(FREQUENCY),)), pulses, (flux,), delay + 2 * width, STRICT
    )
    assert result[1] == pytest.approx((1 + np.cos(shift * delay)) / 2, abs=1e-8)


def test_shaped_flux_ramsey():
    samples = np.array([0.0, 0.03, -0.01, 0.12, 0.0])
    flux = Flux(0, 2, 25, samples)
    pulses = (
        rotation(np.pi / 2, duration=2),
        rotation(np.pi / 2, duration=2, start=27),
    )
    result = evolve(Model((Qubit(FREQUENCY),)), pulses, (flux,), 29, STRICT)
    expected = (1 + np.cos(5 * samples.sum())) / 2
    assert result[1] == pytest.approx(expected, abs=2e-7)


def test_simultaneous_drive_and_flux():
    duration, omega, shift = 34, 0.12, 0.071
    pulse = rotation(omega * duration, duration=duration)
    flux = Flux(0, 0, duration, np.array([shift]))
    result = evolve(Model((Qubit(FREQUENCY),)), (pulse,), (flux,), duration, STRICT)
    rate = np.hypot(omega, shift)
    expected = omega**2 / rate**2 * np.sin(rate * duration / 2) ** 2
    assert result[1] == pytest.approx(expected, abs=1e-8)


def test_overlapping_drives_add_coherently():
    first = rotation(np.pi / 2, duration=20)
    second = rotation(np.pi / 2, duration=20, start=10)
    result = evolve(Model((Qubit(FREQUENCY),)), (second, first), (), 40, STRICT)
    np.testing.assert_allclose(result, [0, 1], atol=1e-8)


def test_exchange():
    strength, width = 12e6, 1e-4
    model = Model((Qubit(FREQUENCY), Qubit(FREQUENCY)), (Coupling(0, 1, strength),))
    pulse = rotation(np.pi, duration=width)
    duration = 1e9 / (4 * strength) + width
    result = evolve(model, (pulse,), (), duration, STRICT)
    np.testing.assert_allclose(result, [0, 1, 0, 0], atol=1e-7)


def test_qutrit_leakage():
    qubit = Qubit(FREQUENCY, anharmonicity=-45e6, levels=3)
    pulse = rotation(np.pi, duration=7)
    a = qt.destroy(3)
    n = a.dag() * a
    h = np.pi * qubit.anharmonicity / 1e9 * n * (n - qt.qeye(3))
    h += pulse.samples[0] * a.dag() + pulse.samples[0].conjugate() * a
    state = (-1j * h * pulse.duration).expm() * qt.basis(3, 0)
    result = evolve(Model((qubit,)), (pulse,), (), pulse.duration, STRICT)
    np.testing.assert_allclose(result, np.abs(state.full().ravel()) ** 2, atol=1e-8)
    assert result[2] > 0.1


def test_short_pulse_after_long_idle():
    pulse = rotation(np.pi, start=1e6, duration=0.01)
    result = evolve(Model((Qubit(FREQUENCY),)), (pulse,), (), 1e6 + 0.01, STRICT)
    np.testing.assert_allclose(result, [0, 1], atol=1e-7)


def test_zero_time():
    model = Model((Qubit(FREQUENCY, levels=3), Qubit(FREQUENCY)))
    np.testing.assert_array_equal(evolve(model, (), (), 0), [1, 0, 0, 0, 0, 0])


def test_frames_and_sparse_cache():
    model = Model(
        tuple(Qubit(FREQUENCY + i * 1e6) for i in range(4)),
        (Coupling(2, 3, 1e6), Coupling(1, 3, 1e6)),
    )
    assert frame_frequencies(model) == (FREQUENCY, *(3 * [FREQUENCY + 1e6]))
    operators = _operators(model)
    assert _operators(model) is operators
    assert _operators.cache_info().maxsize == 16
    assert isinstance(operators[3].data, qt.core.data.CSR)
    dissipative = _operators(Model((Qubit(FREQUENCY, t1=100, t2=100),)))
    assert all(isinstance(c.data, qt.core.data.CSR) for c in dissipative[4])


@pytest.mark.parametrize("coherence", [{}, {"t1": 100}, {"t2": 100}])
def test_solver_selection_and_terminal_storage(monkeypatch, coherence):
    selected = "mesolve" if coherence else "sesolve"
    unused = "sesolve" if coherence else "mesolve"
    solve = getattr(qt, selected)
    calls = []

    def wrapped(*args, **kwargs):
        options = kwargs["options"]
        assert options["store_final_state"]
        assert not options["store_states"]
        assert not options["normalize_output"]
        calls.append(options["max_step"])
        return solve(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail(f"{unused} must not be used")

    monkeypatch.setattr(qt, selected, wrapped)
    monkeypatch.setattr(qt, unused, forbidden)
    model = Model((Qubit(FREQUENCY, **coherence),))
    constant = Drive(0, 100, 20, np.full(20000, -0.01j), FREQUENCY)
    changing = Drive(0, 130, 4, np.array([-0.01j, -0.02j]), FREQUENCY)
    evolve(model, (constant, changing), (), 200, STRICT)
    assert calls == [0, 0, 0, 1, 0]


@pytest.mark.parametrize("diagonal", [[1.1, -0.1], [0.3, 0.3], [np.nan, 1]])
def test_invalid_solver_probabilities(monkeypatch, diagonal):
    def invalid(*args, **kwargs):
        return SimpleNamespace(final_state=qt.Qobj(np.diag(diagonal)))

    monkeypatch.setattr(qt, "mesolve", invalid)
    with pytest.raises(RuntimeError, match="probabilities"):
        evolve(Model((Qubit(FREQUENCY, t1=100),)), (), (), 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frequency": np.nan},
        {"frequency": 0},
        {"anharmonicity": np.inf},
        {"levels": 1},
        {"levels": 2.5},
        {"levels": True},
        {"t1": 0},
        {"t1": np.inf},
        {"t2": -1},
        {"t1": 10, "t2": 21},
    ],
)
def test_invalid_qubits(kwargs):
    with pytest.raises(ValueError):
        Qubit(**({"frequency": FREQUENCY} | kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rtol": 0},
        {"atol": -1},
        {"rtol": np.nan},
        {"nsteps": 0},
        {"nsteps": 1.5},
        {"nsteps": True},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_invalid_models():
    with pytest.raises(ValueError):
        Model(())
    with pytest.raises(ValueError):
        Coupling(0, 0, 1)
    with pytest.raises(ValueError):
        Coupling(0, 1, np.inf)
    with pytest.raises(ValueError):
        Model((Qubit(FREQUENCY),), (Coupling(0, 1, 1),))
    with pytest.raises(ValueError):
        Model(
            (Qubit(FREQUENCY), Qubit(FREQUENCY)), (Coupling(0, 1, 1), Coupling(1, 0, 1))
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": -1},
        {"duration": -1},
        {"duration": np.inf},
        {"samples": np.array([])},
        {"samples": np.array([[1]])},
        {"samples": np.array([np.nan])},
        {"frequency": np.inf},
        {"target": -1},
    ],
)
def test_invalid_drives(kwargs):
    with pytest.raises(ValueError):
        replace(rotation(1), **kwargs)


def test_invalid_evolution():
    model = Model((Qubit(FREQUENCY),))
    for duration in (-1, np.inf, np.nan):
        with pytest.raises(ValueError):
            evolve(model, (), (), duration)
    with pytest.raises(ValueError):
        evolve(model, (rotation(1, target=1),), (), 20)
    with pytest.raises(ValueError):
        evolve(model, (rotation(1),), (), 19)
    with pytest.raises(ValueError):
        Flux(0, 0, 1, np.array([1j]))


def test_samples_are_copied_and_readonly():
    samples = np.array([1j])
    pulse = Drive(0, 0, 1, samples, FREQUENCY)
    samples[0] = 0
    assert pulse.samples[0] == 1j
    assert not pulse.samples.flags.writeable
