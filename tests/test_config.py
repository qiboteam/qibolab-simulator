import pytest
from pydantic import ValidationError
from qibolab import ConfigKinds

from qibolab_simulator import (
    CouplingConfig,
    HamiltonianConfig,
    QubitConfig,
    register_configs,
)


@pytest.mark.parametrize(
    "parameters",
    [
        {"frequency": 0},
        {"frequency": float("nan")},
        {"levels": 1},
        {"t1": -1},
        {"t2": float("inf")},
        {"t1": 1, "t2": 3},
        {"asymmetry": 1.1},
    ],
)
def test_invalid_qubit_config(parameters):
    with pytest.raises(ValidationError):
        QubitConfig(**({"frequency": 5e9} | parameters))


def test_config_registration_idempotent():
    register_configs()
    before = ConfigKinds.registered()
    register_configs()
    assert ConfigKinds.registered() == before


def test_invalid_couplings():
    q = QubitConfig(frequency=5e9)
    with pytest.raises(ValidationError, match="Invalid coupling"):
        HamiltonianConfig(
            qubits={"a": q},
            couplings=(CouplingConfig(qubits=("a", "b"), coupling=1e6),),
        )
    with pytest.raises(ValidationError, match="Duplicate coupling"):
        HamiltonianConfig(
            qubits={"a": q, "b": q},
            couplings=(
                CouplingConfig(qubits=("a", "b"), coupling=1e6),
                CouplingConfig(qubits=("b", "a"), coupling=1e6),
            ),
        )
