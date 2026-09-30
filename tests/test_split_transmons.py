from pathlib import Path

import numpy as np
from qibolab import AveragingMode, PulseSequence, create_platform

from examples.platforms.split_transmons import create_platform as split_transmons
from qibolab_simulator.engine import evolve
from qibolab_simulator.sequence import compile_sequence


def test_platform_loader(monkeypatch):
    directory = Path(__file__).parents[1] / "examples" / "platforms"
    monkeypatch.setenv("QIBOLAB_PLATFORMS", str(directory))
    platform = create_platform("split_transmons")
    model = platform.config("hamiltonian")
    assert platform.nqubits == 2
    assert platform.pairs == [(0, 1)]
    assert [qubit.levels for qubit in model.qubits.values()] == [3, 3]
    assert model.couplings[0].coupling == 12e6
    for target, qubit in platform.qubits.items():
        assert platform.config(qubit.drive).target == target
        assert platform.config(qubit.drive_extra[1, 2]).target == target
        assert platform.config(qubit.flux).offset == model.qubits[target].sweetspot
        assert platform.channels[qubit.acquisition].probe == qubit.probe
        assert qubit.probe in platform.channels
        assert platform.natives.single_qubit[target].RX12 is not None


def test_native_qutrit_preparation():
    platform = split_transmons()
    sequence = PulseSequence()
    for target in platform.qubits:
        sequence += platform.natives.single_qubit[target].RX()
    drives = [q.drive for q in platform.qubits.values()]
    extra_drives = [q.drive_extra[1, 2] for q in platform.qubits.values()]
    acquisitions = [q.acquisition for q in platform.qubits.values()]
    sequence.align(drives + extra_drives)
    for target in platform.qubits:
        sequence += platform.natives.single_qubit[target].RX12()
    sequence.align(drives + extra_drives + acquisitions)
    for target in platform.qubits:
        sequence += platform.natives.single_qubit[target].MZ()
    experiment = compile_sequence(sequence, platform.parameters.configs, 1)
    probabilities = evolve(
        experiment.model,
        experiment.drives,
        experiment.fluxes,
        experiment.duration,
    ).reshape(3, 3)
    assert probabilities.sum(axis=0)[2] > 0.98
    assert probabilities.sum(axis=1)[2] > 0.98


def test_native_flux_gate_execution():
    platform = split_transmons()
    sequence = PulseSequence()
    for target in platform.qubits:
        sequence += platform.natives.single_qubit[target].RX90()
    sequence.align([q.drive for q in platform.qubits.values()] + ["1/flux"])
    sequence += platform.natives.two_qubit[0, 1].CZ()
    sequence.align(
        [q.drive for q in platform.qubits.values()]
        + ["1/flux"]
        + [q.acquisition for q in platform.qubits.values()]
    )
    for target in platform.qubits:
        sequence += platform.natives.single_qubit[target].MZ()
    platform.connect()
    try:
        result = platform.execute([sequence], averaging_mode=AveragingMode.CYCLIC)
    finally:
        platform.disconnect()
    assert len(result) == 2
    assert all(np.isfinite(value) and 0 <= value <= 1 for value in result.values())


def test_platform_instances_do_not_share_updates():
    first, second = split_transmons(), split_transmons()
    first.update({"configs.0/drive.frequency": 4e9})
    assert second.config("0/drive").frequency == 4813586970
