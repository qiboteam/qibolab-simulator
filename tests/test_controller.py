from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from pydantic import ValidationError
from qibolab import (
    Acquisition,
    AcquisitionChannel,
    AcquisitionType,
    AveragingMode,
    Delay,
    ExecutionParameters,
    IqChannel,
    Parameter,
    Parameters,
    Platform,
    PulseSequence,
    Qubit,
    Sweeper,
)

from qibolab_simulator import SimulatorController, register_configs


def execute(configs, sequence, sweepers=None, **kwargs):
    controller = SimulatorController(seed=71)
    options = ExecutionParameters(averaging_mode=AveragingMode.CYCLIC, **kwargs)
    return controller.play(configs, [sequence], options, sweepers or [])


@pytest.mark.parametrize("mode", list(AveragingMode))
@pytest.mark.parametrize(
    "acquisition", [AcquisitionType.INTEGRATION, AcquisitionType.DISCRIMINATION]
)
def test_shapes_and_rabi(configs, experiment, mode, acquisition):
    sequence, pulse, readout = experiment
    amplitude = Sweeper(
        parameter=Parameter.amplitude, pulses=[pulse], values=np.array([0, 0.5, 1])
    )
    frequency = Sweeper(
        parameter=Parameter.frequency,
        channels=["microwave"],
        values=np.array([5e9, 5e9]),
    )
    options = ExecutionParameters(
        nshots=64, averaging_mode=mode, acquisition_type=acquisition
    )
    values = SimulatorController(seed=10).play(
        configs, [sequence], options, [[amplitude], [frequency]]
    )[readout.id]
    assert values.shape == options.results_shape([[amplitude], [frequency]])
    if acquisition is AcquisitionType.INTEGRATION:
        assert np.all(values[..., 1] == 0)
        values = values[..., 0]
    if mode is AveragingMode.SINGLESHOT:
        assert np.all(values[:, 0] == 0)
        assert np.all(values[:, -1] == 1)
    else:
        np.testing.assert_allclose(values, [[0, 0], [0.5, 0.5], [1, 1]], atol=2e-6)
    assert pulse.amplitude == 1
    assert configs["microwave"].frequency == 5e9


def test_duration_sweep_recomputes_align(configs, experiment):
    sequence, pulse, readout = experiment
    sweep = Sweeper(
        parameter=Parameter.duration, pulses=[pulse], values=np.array([0, 20, 40])
    )
    result = execute(configs, sequence, [[sweep]])
    np.testing.assert_allclose(result[readout.id], [0, 0.5, 1], atol=2e-6)
    assert pulse.duration == 40


def test_parallel_axes_are_zipped(configs, experiment):
    sequence, pulse, readout = experiment
    sweeps = [
        [
            Sweeper(
                parameter=Parameter.amplitude,
                pulses=[pulse],
                values=np.array([0, 0.5, 1]),
            ),
            Sweeper(
                parameter=Parameter.duration_interpolated,
                pulses=[pulse],
                values=np.array([20, 40, 20]),
            ),
        ]
    ]
    np.testing.assert_allclose(
        execute(configs, sequence, sweeps)[readout.id], [0, 0.5, 0.5], atol=2e-6
    )


def test_delay_duration_sweep(configs, experiment):
    from qibolab_simulator import HamiltonianConfig, QubitConfig

    _, pulse, readout = experiment
    configs["hamiltonian"] = HamiltonianConfig(
        qubits={"alice": QubitConfig(frequency=5e9, t1=100)}
    )
    delay = Delay(duration=0)
    sequence = PulseSequence([("microwave", pulse), ("microwave", delay)])
    sequence.align(["microwave", "detector"])
    sequence.append(("detector", readout))
    sweep = Sweeper(
        parameter=Parameter.duration, pulses=[delay], values=np.array([0, 100, 200])
    )
    value = execute(configs, sequence, [[sweep]])[readout.id]
    np.testing.assert_allclose(value / value[0], np.exp(-np.arange(3)), atol=2e-6)


def test_long_qutrit_delay_sweep(configs, experiment):
    from qibolab_simulator import HamiltonianConfig, QubitConfig

    _, pulse, readout = experiment
    configs["hamiltonian"] = HamiltonianConfig(
        qubits={"alice": QubitConfig(frequency=5e9, anharmonicity=-200e6, levels=3)}
    )
    delay = Delay(duration=0)
    sequence = PulseSequence([("microwave", pulse), ("microwave", delay)])
    sequence.align(["microwave", "detector"])
    sequence.append(("detector", readout))
    sweep = Sweeper(
        parameter=Parameter.duration,
        pulses=[delay],
        values=np.array([0, 1e4, 1e6]),
    )
    value = execute(configs, sequence, [[sweep]])[readout.id]
    np.testing.assert_allclose(value, value[0], atol=1e-12)


def test_multiple_sequences_and_empty(configs, experiment):
    sequence, _, readout = experiment
    second = Acquisition(duration=1)
    controller = SimulatorController()
    options = ExecutionParameters(averaging_mode=AveragingMode.CYCLIC)
    result = controller.play(
        configs, [sequence, PulseSequence([("detector", second)])], options, []
    )
    assert result[readout.id] == pytest.approx(1, abs=2e-6)
    assert result[second.id] == 0
    assert controller.play(configs, [], options, []) == {}
    with pytest.raises(ValueError, match="unique"):
        controller.play(configs, [sequence, sequence], options, [])


def test_platform_execute_and_runcard(configs, experiment):
    register_configs()
    parameters = Parameters(configs=configs)
    loaded = Parameters.model_validate_json(parameters.model_dump_json())
    controller = SimulatorController(
        channels={
            "microwave": IqChannel(),
            "detector": AcquisitionChannel(),
        }
    )
    platform = Platform(
        name="test",
        parameters=loaded,
        instruments={"simulator": controller},
        qubits={"alice": Qubit(drive="microwave", acquisition="detector")},
    )
    sequence, _, readout = experiment
    platform.connect()
    try:
        result = platform.execute(
            [sequence],
            averaging_mode=AveragingMode.CYCLIC,
            updates=[{"microwave": {"rabi_frequency": 6.25e6}}],
        )
    finally:
        platform.disconnect()
    assert result[readout.id] == pytest.approx(0.5, abs=2e-6)
    assert platform.parameters.configs["microwave"].rabi_frequency == 12.5e6


@pytest.mark.slow
def test_serial_process_reproducibility(configs, experiment):
    sequence, pulse, readout = experiment
    sweeper = Sweeper(
        parameter=Parameter.amplitude, pulses=[pulse], values=np.linspace(0, 1, 9)
    )
    options = ExecutionParameters(nshots=60)
    serial = SimulatorController(seed=126).play(
        configs, [sequence], options, [[sweeper]]
    )
    controller = SimulatorController(seed=126, workers=2, batch_size=2, max_in_flight=2)
    controller.connect()
    pool = controller._pool
    try:
        parallel = controller.play(configs, [sequence], options, [[sweeper]])
        np.testing.assert_array_equal(serial[readout.id], parallel[readout.id])
        assert controller._pool is pool
    finally:
        controller.disconnect()
    assert controller._pool is None


@pytest.mark.parametrize(
    "options,match",
    [
        (ExecutionParameters(acquisition_type=AcquisitionType.RAW), "RAW"),
        (ExecutionParameters(fast_reset=True), "reset"),
        (ExecutionParameters(), "positive nshots"),
        (ExecutionParameters(nshots=0), "positive nshots"),
    ],
)
def test_unsupported_options(configs, experiment, options, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        SimulatorController().play(configs, [experiment[0]], options, [])


def test_thread_executor_rejected():
    with (
        ThreadPoolExecutor(1) as executor,
        pytest.raises(ValidationError, match="process executor"),
    ):
        SimulatorController(executor=executor)
