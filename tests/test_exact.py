"""Exact piecewise-constant propagation against strict adaptive references."""

import numpy as np
import pytest
import qutip as qt

from qibolab_simulator import engine
from qibolab_simulator.engine import (
    Coupling,
    Drive,
    Flux,
    Model,
    Qubit,
    Settings,
    evolve,
)

FREQUENCY = 5e9
REFERENCE = Settings(rtol=1e-11, atol=1e-13)
EXACT = Settings(propagation="exact")


def rotation(angle, *, start=0, duration=20, phase=0, frequency=FREQUENCY, target=0):
    return Drive(
        target,
        start,
        duration,
        np.array([-1j * angle / (2 * duration) * np.exp(-1j * phase)]),
        frequency,
    )


def gaussian(start, duration, frequency, target=0, angle=np.pi):
    count = int(duration)
    envelope = np.exp(-0.5 * ((np.arange(count) + 0.5 - count / 2) / (count / 5)) ** 2)
    samples = -1j * angle * envelope / (2 * envelope.sum())
    return Drive(target, start, duration, samples, frequency)


QUTRITS = Model(
    (
        Qubit(4.8e9, -217e6, 3, t1=3700.0, t2=2300.0),
        Qubit(5.5e9, -212e6, 3, t1=3200.0, t2=900.0),
    ),
    (Coupling(0, 1, 12e6),),
)
CLOSED_QUTRITS = Model(
    tuple(Qubit(q.frequency, q.anharmonicity, q.levels) for q in QUTRITS.qubits),
    QUTRITS.couplings,
)
UNCOUPLED = Model(QUTRITS.qubits)
SINGLE = Model((Qubit(FREQUENCY),))
DAMPED = Model((Qubit(FREQUENCY, t1=130.0, t2=90.0),))

CASES = {
    "resonant": (SINGLE, (rotation(np.pi / 3),), (), 20.0),
    "detuned": (SINGLE, (rotation(1.7, frequency=FREQUENCY + 23e6),), (), 25.0),
    "shaped": (SINGLE, (gaussian(7, 30, FREQUENCY + 4e6),), (), 41.0),
    "phases": (
        SINGLE,
        (rotation(np.pi / 2), rotation(np.pi / 2, start=33, phase=0.7)),
        (),
        53.0,
    ),
    "overlapping": (
        SINGLE,
        (
            gaussian(0, 20, FREQUENCY + 3e6),
            rotation(1.1, start=10, duration=20, frequency=FREQUENCY + 3e6),
        ),
        (),
        30.0,
    ),
    "damped": (DAMPED, (rotation(np.pi, duration=3),), (), 83.0),
    "ramsey": (
        DAMPED,
        (
            rotation(np.pi / 2, duration=4, frequency=FREQUENCY + 9e6),
            rotation(np.pi / 2, duration=4, start=61, frequency=FREQUENCY + 9e6),
        ),
        (),
        65.0,
    ),
    "qubit 1": (QUTRITS, (gaussian(0, 40, 5.5e9 + 1e6, target=1),), (), 40.0),
    "rx then rx12": (
        QUTRITS,
        (gaussian(0, 24, 4.8e9), gaussian(24, 24, 4.583e9)),
        (),
        48.0,
    ),
    "drive and flux": (
        QUTRITS,
        (gaussian(0, 24, 4.8e9 + 2e6),),
        (
            Flux(1, 10, 40, -2 * np.pi * 0.3 * np.sin(np.linspace(0, np.pi, 40))),
            Flux(0, 70, 20, np.full(20, -0.05)),
        ),
        130.0,
    ),
    "closed": (
        CLOSED_QUTRITS,
        (gaussian(0, 24, 4.8e9 + 3e6), gaussian(30, 24, 4.8e9 + 3e6, target=1)),
        (Flux(1, 60, 30, np.full(30, -0.4)),),
        90.0,
    ),
    "uncoupled detunings": (
        UNCOUPLED,
        (gaussian(0, 24, 4.8e9 + 2e6), gaussian(0, 24, 5.5e9 - 3e6, target=1)),
        (),
        24.0,
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_agrees_with_reference(name):
    problem = CASES[name]
    np.testing.assert_allclose(
        evolve(*problem, EXACT), evolve(*problem, REFERENCE), atol=1e-9
    )


@pytest.mark.parametrize("name", ["shaped", "ramsey", "drive and flux", "closed"])
def test_sparse_action_agrees_with_dense(monkeypatch, name):
    problem = CASES[name]
    dense = evolve(*problem, EXACT)
    monkeypatch.setattr(engine, "_DENSE_LIMIT", 0)
    np.testing.assert_allclose(evolve(*problem, EXACT), dense, atol=1e-12)


@pytest.mark.parametrize("detuning", [-80e6, -5e6, 0, 5e6, 80e6])
def test_detuned_rabi(detuning):
    duration, rabi = 63.0, 13e6
    omega, delta = 2 * np.pi * rabi / 1e9, 2 * np.pi * detuning / 1e9
    pulse = rotation(
        omega * duration, duration=duration, frequency=FREQUENCY + detuning
    )
    result = evolve(SINGLE, (pulse,), (), duration, EXACT)
    rate = np.hypot(omega, delta)
    expected = omega**2 / rate**2 * np.sin(rate * duration / 2) ** 2
    assert result[1] == pytest.approx(expected, abs=1e-12)


def solver_calls(monkeypatch):
    calls = []
    for name in ("sesolve", "mesolve"):
        solve = getattr(qt, name)

        def wrapped(*args, solve=solve, name=name, **kwargs):
            calls.append(name)
            return solve(*args, **kwargs)

        monkeypatch.setattr(qt, name, wrapped)
    return calls


@pytest.mark.parametrize("name", list(CASES))
def test_no_adaptive_solver(monkeypatch, name):
    calls = solver_calls(monkeypatch)
    evolve(*CASES[name], EXACT)
    assert calls == []


def test_different_detunings_in_a_component_fall_back(monkeypatch):
    drives = (gaussian(0, 24, 4.8e9), gaussian(12, 24, 5.5e9, target=1))
    calls = solver_calls(monkeypatch)
    result = evolve(QUTRITS, drives, (), 60.0, EXACT)
    # Only the overlap falls back; leading, trailing, and idle segments are exact.
    assert len(calls) == 1
    np.testing.assert_allclose(
        result, evolve(QUTRITS, drives, (), 60.0, REFERENCE), atol=1e-6
    )


def test_constant_detuned_pulse_is_one_exponential(monkeypatch):
    exponentials = []

    def counted(matrix):
        exponentials.append(matrix)
        return expm(matrix)

    expm = engine.expm
    monkeypatch.setattr(engine, "expm", counted)
    pulse = Drive(0, 0, 1000, np.full(1000, -0.003j), 4.8e9 + 7e6)
    evolve(QUTRITS, (pulse,), (), 1000.0, EXACT)
    assert len(exponentials) == 1


def test_symmetric_envelope_reuses_exponentials(monkeypatch):
    exponentials = []

    def counted(matrix):
        exponentials.append(matrix)
        return expm(matrix)

    expm = engine.expm
    monkeypatch.setattr(engine, "expm", counted)
    evolve(SINGLE, (gaussian(0, 40, FREQUENCY),), (), 40.0, EXACT)
    assert len(exponentials) == 20


def test_long_detuned_idle_ramsey():
    delay, detuning = 1e3, 2e6
    pulses = (
        rotation(np.pi / 2, duration=4, frequency=FREQUENCY + detuning),
        rotation(
            np.pi / 2, start=4 + delay, duration=4, frequency=FREQUENCY + detuning
        ),
    )
    np.testing.assert_allclose(
        evolve(DAMPED, pulses, (), delay + 8, EXACT),
        evolve(DAMPED, pulses, (), delay + 8, REFERENCE),
        atol=1e-9,
    )


def test_invalid_propagation():
    with pytest.raises(ValueError, match="propagation"):
        Settings(propagation="euler")


@pytest.mark.slow
def test_qutrit_platform_sweeps():
    """Detuned natives, delay and flux-offset sweeps, through the controller."""
    from qibolab import (
        AveragingMode,
        Delay,
        ExecutionParameters,
        Parameter,
        PulseSequence,
        Sweeper,
    )

    from examples.platforms.qutrits import create_platform
    from qibolab_simulator import SimulatorController

    platform = create_platform()
    natives = platform.natives.single_qubit
    delay = Delay(duration=0)
    sequence = PulseSequence()
    sequence += natives[0].RX90()
    sequence.append(("0/drive", delay))
    sequence += natives[0].RX90()
    # Simultaneous drives on the coupled pair would fall back to adaptive solving.
    sequence.align(["0/drive", "1/drive"])
    sequence += natives[1].RX()
    sequence.align(["0/drive", "1/drive", "0/acquisition", "1/acquisition"])
    sequence += natives[0].MZ() + natives[1].MZ()
    sweepers = [
        [
            Sweeper(
                parameter=Parameter.duration,
                pulses=[delay],
                values=np.arange(0, 400, 100.0),
            )
        ],
        [
            Sweeper(
                parameter=Parameter.frequency,
                channels=["0/drive"],
                values=platform.config("0/drive").frequency + np.array([-2e6, 3e6]),
            )
        ],
        [
            Sweeper(
                parameter=Parameter.offset,
                channels=["1/flux"],
                values=0.582 + np.array([0.0, 0.02]),
            )
        ],
    ]
    options = ExecutionParameters(nshots=1, averaging_mode=AveragingMode.CYCLIC)
    configs = platform.parameters.configs
    results = [
        SimulatorController(seed=17, solver=solver).play(
            configs, [sequence], options, sweepers
        )
        for solver in (REFERENCE, EXACT)
    ]
    assert results[0].keys() == results[1].keys()
    for key in results[0]:
        np.testing.assert_allclose(results[1][key], results[0][key], atol=1e-9)
