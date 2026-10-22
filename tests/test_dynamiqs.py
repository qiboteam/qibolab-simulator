"""The dynamiqs backend against the QuTiP engine, on identical problems."""

import numpy as np
import pytest

pytest.importorskip("dynamiqs")

from qibolab_simulator.dynamiqs_engine import (  # noqa: E402
    _bucket,
    _segments,
    evolve_batch,
)
from qibolab_simulator.engine import (  # noqa: E402
    Coupling,
    Drive,
    DynamiqsSettings,
    Flux,
    Model,
    Qubit,
    Settings,
    evolve,
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


def gaussian(start, duration, frequency, target=0, angle=np.pi, count=None):
    count = count or int(duration)
    envelope = np.exp(-0.5 * ((np.arange(count) + 0.5 - count / 2) / (count / 5)) ** 2)
    samples = -1j * angle * envelope / (2 * envelope.sum() * duration / count)
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
SINGLE = Model((Qubit(FREQUENCY),))
DAMPED = Model((Qubit(FREQUENCY, t1=130.0, t2=90.0),))

CASES = {
    "idle": (SINGLE, (), (), 30.0),
    "empty": (SINGLE, (), (), 0.0),
    "resonant": (SINGLE, (rotation(np.pi / 3),), (), 20.0),
    "detuned": (SINGLE, (rotation(1.7, frequency=FREQUENCY + 23e6),), (), 25.0),
    "shaped": (SINGLE, (gaussian(7, 30, FREQUENCY + 4e6),), (), 41.0),
    "phases": (
        SINGLE,
        (rotation(np.pi / 2), rotation(np.pi / 2, start=33, phase=0.7)),
        (),
        53.0,
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
    "qutrits": (
        QUTRITS,
        (
            gaussian(0, 24, 4.8e9, target=0),
            gaussian(10, 32, 5.5e9, target=1, angle=np.pi / 2),
            gaussian(30, 24, 4.583e9, target=0, angle=np.pi / 2),
        ),
        (
            Flux(1, 60, 40, -2 * np.pi * 0.3 * np.sin(np.linspace(0, np.pi, 40))),
            Flux(0, 70, 20, np.full(20, -0.05)),
        ),
        130.0,
    ),
    "closed-qutrits": (
        CLOSED_QUTRITS,
        (gaussian(0, 24, 4.8e9), gaussian(5, 24, 5.5e9, target=1)),
        (Flux(1, 40, 30, np.full(30, -0.4)),),
        90.0,
    ),
}


# Coupled qutrits compile large batched solvers.
SLOW_CASES = {"qutrits", "closed-qutrits", "ramsey", "detuned qubit 1"}


def cases(names):
    return [
        pytest.param(name, marks=pytest.mark.slow) if name in SLOW_CASES else name
        for name in names
    ]


@pytest.mark.parametrize("name", cases(CASES))
def test_single_problem(name):
    problem = CASES[name]
    (result,) = evolve_batch([problem], STRICT)
    np.testing.assert_allclose(result, evolve(*problem, STRICT), atol=1e-7)


FOLLOW = DynamiqsSettings(frame="drives")
REFERENCE = Settings(rtol=1e-11, atol=1e-13)
FOLLOW_CASES = CASES | {
    "constant detuned": (
        QUTRITS,
        (Drive(0, 10, 1000, np.full(1000, -0.003j), 4.8e9 + 7e6),),
        (),
        1020.0,
    ),
    "detuned qubit 1": (QUTRITS, (gaussian(0, 40, 5.5e9 + 1e6, target=1),), (), 40.0),
    "closed detuned": (
        CLOSED_QUTRITS,
        (rotation(1.3, start=5, duration=40, frequency=4.8e9 - 9e6),),
        (),
        50.0,
    ),
}


@pytest.mark.parametrize("name", cases(FOLLOW_CASES))
def test_frame_following_drives(name):
    problem = FOLLOW_CASES[name]
    (result,) = evolve_batch([problem], STRICT, FOLLOW)
    np.testing.assert_allclose(result, evolve(*problem, REFERENCE), atol=1e-8)


def test_constant_detuned_drives_become_static():
    model, drives, fluxes, duration = FOLLOW_CASES["constant detuned"]
    assert _segments(model, drives, fluxes, duration)[1].terms
    shifted = _segments(model, drives, fluxes, duration, follow=True)
    assert [s.terms for s in shifted] == [(), (), ()]
    assert np.any(shifted[1].frame) and not np.any(shifted[0].frame)
    # Different detunings in one component keep the fixed frame.
    drives = (rotation(1, frequency=4.8e9), rotation(1, frequency=5.5e9, target=1))
    (segment,) = _segments(QUTRITS, drives, (), 20.0, follow=True)
    assert len(segment.terms) == 1 and not np.any(segment.frame)


@pytest.mark.slow
def test_spectroscopy_sweep_with_frame_following():
    frequencies = 4.8e9 + np.linspace(-20e6, 20e6, 9)
    problems = [
        (
            QUTRITS,
            (Drive(0, 0, 1000, np.full(1, -0.002j), frequency),),
            (),
            1000.0,
        )
        for frequency in frequencies
    ]
    results = evolve_batch(problems, STRICT, FOLLOW)
    for problem, result in zip(problems, results, strict=True):
        np.testing.assert_allclose(result, evolve(*problem, REFERENCE), atol=1e-8)


@pytest.mark.slow
def test_mixed_batch_preserves_order():
    problems = list(CASES.values()) * 2
    results = evolve_batch(problems, STRICT)
    for problem, result in zip(problems, results, strict=True):
        np.testing.assert_allclose(result, evolve(*problem, STRICT), atol=1e-7)


@pytest.mark.slow
def test_sweeps_within_one_structure():
    """Amplitudes, frequencies, durations, and biases differ within each group."""
    problems = []
    for amplitude in np.linspace(0, 2, 5):
        for detuning in (-15e6, 0.0, 15e6):
            pulse = rotation(amplitude * np.pi, frequency=FREQUENCY + detuning)
            problems.append((DAMPED, (pulse,), (), 20.0 + 10 * amplitude))
    for duration in (8, 13, 21, 34):
        problems.append((SINGLE, (gaussian(2, duration, FREQUENCY + 1e6),), (), 40.0))
    for frequency in (5.45e9, 5.5e9, 5.55e9):
        model = Model(
            (QUTRITS.qubits[0], Qubit(frequency, -212e6, 3, t1=3200.0, t2=900.0)),
            QUTRITS.couplings,
        )
        problems.append((model, (gaussian(0, 16, 5.5e9, target=1),), (), 300.0))
    results = evolve_batch(problems, STRICT)
    for problem, result in zip(problems, results, strict=True):
        np.testing.assert_allclose(result, evolve(*problem, STRICT), atol=1e-7)


@pytest.mark.slow
def test_single_precision():
    loose = Settings(rtol=1e-5, atol=1e-6)
    problems = [CASES["shaped"], CASES["qutrits"]]
    results = evolve_batch(problems, loose, DynamiqsSettings(precision="single"))
    for problem, result in zip(problems, results, strict=True):
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, evolve(*problem, STRICT), atol=1e-4)
    # Leave the process-wide default as the other tests expect it.
    evolve_batch([CASES["idle"]], STRICT)


@pytest.mark.parametrize("method", ["dopri5", "tsit5"])
def test_methods(method):
    problem = CASES["detuned"]
    (result,) = evolve_batch([problem], STRICT, DynamiqsSettings(method=method))
    np.testing.assert_allclose(result, evolve(*problem, STRICT), atol=1e-7)


def test_shared_validation():
    pulse = rotation(np.pi, duration=40)
    with pytest.raises(ValueError, match="beyond evolution"):
        evolve_batch([(SINGLE, (pulse,), (), 20.0)])
    with pytest.raises(ValueError, match="DynamiqsSettings"):
        evolve_batch([CASES["idle"]], STRICT, Settings())
    with pytest.raises(ValueError, match="method"):
        DynamiqsSettings(method="euler")
    with pytest.raises(ValueError, match="frame"):
        DynamiqsSettings(frame="lab")


def test_bucket():
    assert [_bucket(n) for n in (0, 1, 2, 3, 4, 5, 1000)] == [1, 1, 2, 4, 4, 8, 1024]


def play(backend, configs, sequences, sweepers, **kwargs):
    from qibolab import ExecutionParameters

    from qibolab_simulator import SimulatorController

    controller = SimulatorController(seed=17, backend=backend, solver=STRICT)
    options = ExecutionParameters(**kwargs)
    return controller.play(configs, sequences, options, sweepers)


@pytest.mark.parametrize("mode", ["CYCLIC", "SINGLESHOT"])
def test_controller_backends_agree(configs, experiment, mode):
    from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

    sequence, pulse, readout = experiment
    sweepers = [
        [
            Sweeper(
                parameter=Parameter.amplitude,
                pulses=[pulse],
                values=np.linspace(0, 1, 7),
            )
        ],
        [
            Sweeper(
                parameter=Parameter.frequency,
                channels=["microwave"],
                values=np.linspace(4.98e9, 5.02e9, 5),
            )
        ],
    ]
    kwargs = dict(
        nshots=200,
        averaging_mode=AveragingMode[mode],
        acquisition_type=AcquisitionType.INTEGRATION,
    )
    qutip = play("qutip", configs, [sequence], sweepers, **kwargs)[readout.id]
    dynamiqs = play("dynamiqs", configs, [sequence], sweepers, **kwargs)[readout.id]
    assert qutip.shape == dynamiqs.shape
    if mode == "CYCLIC":
        np.testing.assert_allclose(dynamiqs, qutip, atol=1e-7)
    else:
        # Equal seeds and nearly equal probabilities give nearly identical shots.
        assert np.mean(dynamiqs == qutip) > 0.999


@pytest.mark.slow
def test_qutrit_platform_sweeps():
    """Natives, flux-offset and delay sweeps, with dissipation and coupling."""
    from qibolab import AveragingMode, Delay, Parameter, PulseSequence, Sweeper

    from examples.platforms.qutrits import create_platform

    platform = create_platform()
    configs = platform.parameters.configs
    natives = platform.natives.single_qubit
    delay = Delay(duration=0)
    sequence = PulseSequence()
    sequence += natives[0].RX90()
    sequence += natives[1].RX()
    sequence.append(("0/drive", delay))
    sequence += natives[0].RX90()
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
                parameter=Parameter.offset,
                channels=["1/flux"],
                values=0.582 + np.array([0.0, 0.02]),
            )
        ],
    ]
    kwargs = dict(nshots=1, averaging_mode=AveragingMode.CYCLIC)
    qutip = play("qutip", configs, [sequence], sweepers, **kwargs)
    dynamiqs = play("dynamiqs", configs, [sequence], sweepers, **kwargs)
    assert qutip.keys() == dynamiqs.keys()
    for key in qutip:
        np.testing.assert_allclose(dynamiqs[key], qutip[key], atol=1e-7)


def test_batch_size_defaults():
    from qibolab_simulator import SimulatorController

    assert SimulatorController().effective_batch_size == 8
    assert SimulatorController(backend="dynamiqs").effective_batch_size == 1024
    assert (
        SimulatorController(backend="dynamiqs", batch_size=3).effective_batch_size == 3
    )


@pytest.mark.slow
def test_process_workers(configs, experiment):
    from qibolab import ExecutionParameters, Parameter, Sweeper

    from qibolab_simulator import SimulatorController

    sequence, pulse, readout = experiment
    sweeper = Sweeper(
        parameter=Parameter.amplitude, pulses=[pulse], values=np.linspace(0, 1, 9)
    )
    options = ExecutionParameters(nshots=60)
    serial = SimulatorController(seed=5, backend="dynamiqs").play(
        configs, [sequence], options, [[sweeper]]
    )
    parallel = SimulatorController(
        seed=5, backend="dynamiqs", workers=2, batch_size=4
    ).play(configs, [sequence], options, [[sweeper]])
    np.testing.assert_array_equal(serial[readout.id], parallel[readout.id])
