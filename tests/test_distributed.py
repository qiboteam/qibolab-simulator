import numpy as np
import pytest
from qibolab import ExecutionParameters, Parameter, Sweeper

from qibolab_simulator import SimulatorController

distributed = pytest.importorskip("distributed")


@pytest.mark.slow
def test_dask_executor_matches_serial(configs, experiment):
    sequence, pulse, readout = experiment
    sweeper = Sweeper(
        parameter=Parameter.amplitude, pulses=[pulse], values=np.linspace(0, 1, 7)
    )
    options = ExecutionParameters(nshots=32)
    serial = SimulatorController(seed=123).play(
        configs, [sequence], options, [[sweeper]]
    )
    with distributed.LocalCluster(
        n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None
    ) as cluster:
        with distributed.Client(cluster) as client:
            with client.get_executor() as executor:
                controller = SimulatorController(
                    executor=executor, seed=123, batch_size=2, max_in_flight=4
                )
                parallel = controller.play(configs, [sequence], options, [[sweeper]])
                controller.disconnect()
                np.testing.assert_array_equal(serial[readout.id], parallel[readout.id])
                assert executor.submit(abs, -3).result() == 3
