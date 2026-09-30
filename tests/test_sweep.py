import numpy as np
import pytest
from qibolab import Parameter, Sweeper

from qibolab_simulator.sweep import points, sweep_shape


def test_invalid_sweep_groups(experiment):
    _, pulse, _ = experiment
    a = Sweeper(parameter=Parameter.duration, pulses=[pulse], values=np.array([10, 20]))
    b = Sweeper(
        parameter=Parameter.relative_phase, pulses=[pulse], values=np.array([0])
    )
    with pytest.raises(ValueError, match="equal nonzero"):
        sweep_shape([[a, b]])
    with pytest.raises(ValueError, match="empty"):
        sweep_shape([[]])


def test_unknown_sweep_target(configs, experiment):
    sequence, pulse, _ = experiment
    sweeper = Sweeper(
        parameter=Parameter.amplitude, pulses=[pulse.new()], values=np.array([0.5])
    )
    with pytest.raises(ValueError, match="Invalid pulse"):
        list(points([sequence], configs, [[sweeper]]))


def test_duplicate_sweep_target(configs, experiment):
    sequence, pulse, _ = experiment
    sweeper = Sweeper(
        parameter=Parameter.amplitude, pulses=[pulse], values=np.array([0.5])
    )
    with pytest.raises(ValueError, match="Duplicate sweep"):
        list(points([sequence], configs, [[sweeper], [sweeper]]))


def test_repeated_target_in_one_sweeper(configs, experiment):
    sequence, pulse, _ = experiment
    sweeper = Sweeper(
        parameter=Parameter.amplitude,
        pulses=[pulse, pulse],
        values=np.array([0.5]),
    )
    [(_, _, updated, _)] = points([sequence], configs, [[sweeper]])
    assert updated[0][1].amplitude == 0.5
    assert pulse.amplitude == 1


def test_invalid_config_update_is_validated(configs, experiment):
    sweeper = Sweeper(
        parameter=Parameter.frequency, channels=["microwave"], values=np.array([-1])
    )
    with pytest.raises(ValueError):
        list(points([experiment[0]], configs, [[sweeper]]))
