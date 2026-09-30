from uuid import uuid4

import numpy as np
from qibolab import AveragingMode, ExecutionParameters

from qibolab_simulator.controller import _measure
from qibolab_simulator.engine import Model, Qubit
from qibolab_simulator.sequence import Experiment


def test_entangled_joint_shots_and_heterogeneous_levels():
    a, b = uuid4(), uuid4()
    experiment = Experiment(
        Model((Qubit(5e9), Qubit(5e9, levels=3))),
        (),
        (),
        0,
        ((a, 0), (b, 1)),
    )
    probabilities = np.array([0.5, 0, 0, 0, 0, 0.5])
    result = _measure(
        probabilities,
        experiment,
        ExecutionParameters(nshots=1000),
        np.random.default_rng(42),
    )
    np.testing.assert_array_equal(result[a], result[b])
    assert set(result[a]) == {0, 1}
    averaged = _measure(
        probabilities,
        experiment,
        ExecutionParameters(averaging_mode=AveragingMode.CYCLIC),
        np.random.default_rng(42),
    )
    assert averaged[a] == averaged[b] == 0.5


def test_same_target_measurements_share_shots():
    a, b = uuid4(), uuid4()
    experiment = Experiment(Model((Qubit(5e9),)), (), (), 0, ((a, 0), (b, 0)))
    result = _measure(
        np.array([0.3, 0.7]),
        experiment,
        ExecutionParameters(nshots=100),
        np.random.default_rng(42),
    )
    np.testing.assert_array_equal(result[a], result[b])
