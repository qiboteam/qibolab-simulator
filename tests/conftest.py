import pytest
from qibolab import Acquisition, Pulse, PulseSequence, Rectangular

from qibolab_simulator import DriveConfig, HamiltonianConfig, QubitConfig, ReadoutConfig


@pytest.fixture
def configs():
    return {
        "hamiltonian": HamiltonianConfig(qubits={"alice": QubitConfig(frequency=5e9)}),
        "microwave": DriveConfig(target="alice", frequency=5e9, rabi_frequency=12.5e6),
        "detector": ReadoutConfig(target="alice"),
    }


@pytest.fixture
def experiment():
    pulse = Pulse(duration=40, amplitude=1, envelope=Rectangular())
    readout = Acquisition(duration=100)
    sequence = PulseSequence([("microwave", pulse)])
    sequence.align(["microwave", "detector"])
    sequence.append(("detector", readout))
    return sequence, pulse, readout
