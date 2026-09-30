"""A minimal Qibolab platform and a Rabi amplitude sweep."""

import numpy as np
from qibolab import (
    Acquisition,
    AcquisitionChannel,
    AveragingMode,
    IqChannel,
    Parameter,
    Parameters,
    Platform,
    Pulse,
    PulseSequence,
    Qubit,
    Rectangular,
    Sweeper,
)

from qibolab_simulator import (
    DriveConfig,
    HamiltonianConfig,
    QubitConfig,
    ReadoutConfig,
    SimulatorController,
    register_configs,
)


def create_platform(workers: int = 1) -> Platform:
    register_configs()
    controller = SimulatorController(
        workers=workers,
        seed=1234,
        channels={
            "drive": IqChannel(),
            "readout": AcquisitionChannel(),
        },
    )
    return Platform(
        name="simulator",
        instruments={"simulator": controller},
        qubits={"q": Qubit(drive="drive", acquisition="readout")},
        parameters=Parameters(
            configs={
                "hamiltonian": HamiltonianConfig(
                    qubits={"q": QubitConfig(frequency=5e9)}
                ),
                "drive": DriveConfig(target="q", frequency=5e9, rabi_frequency=12.5e6),
                "readout": ReadoutConfig(target="q"),
            }
        ),
    )


def rabi():
    pulse = Pulse(duration=40, amplitude=1, envelope=Rectangular())
    readout = Acquisition(duration=100)
    sequence = PulseSequence([("drive", pulse)])
    sequence.align(["drive", "readout"])
    sequence.append(("readout", readout))
    sweeper = Sweeper(
        parameter=Parameter.amplitude,
        pulses=[pulse],
        values=np.linspace(0, 1, 21),
    )
    return sequence, readout, sweeper


if __name__ == "__main__":
    platform = create_platform()
    sequence, readout, sweeper = rabi()
    platform.connect()
    try:
        results = platform.execute(
            [sequence], [[sweeper]], averaging_mode=AveragingMode.CYCLIC
        )
        print(results[readout.id])
    finally:
        platform.disconnect()
