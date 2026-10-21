Quickstart
==========

A simulated platform is an ordinary Qibolab :class:`~qibolab.Platform`, whose
instrument is a :class:`~qibolab_simulator.SimulatorController`. Its parameters
contain the physical model, under the ``"hamiltonian"`` key, and one simulator
configuration per channel, each targeting a qubit explicitly. Channel and qubit
names are arbitrary.

.. code-block:: python

    from qibolab import (
        Acquisition,
        AcquisitionChannel,
        AveragingMode,
        IqChannel,
        Parameters,
        Platform,
        Pulse,
        PulseSequence,
        Qubit,
        Rectangular,
    )

    from qibolab_simulator import (
        DriveConfig,
        HamiltonianConfig,
        QubitConfig,
        ReadoutConfig,
        SimulatorController,
        register_configs,
    )

    register_configs()  # Also needed before loading serialized simulator configs.
    controller = SimulatorController(
        seed=1234, channels={"xy": IqChannel(), "adc": AcquisitionChannel()}
    )
    platform = Platform(
        name="one-qubit",
        instruments={"simulator": controller},
        qubits={"q": Qubit(drive="xy", acquisition="adc")},
        parameters=Parameters(
            configs={
                "hamiltonian": HamiltonianConfig(
                    qubits={"q": QubitConfig(frequency=5e9, levels=2)}
                ),
                "xy": DriveConfig(target="q", frequency=5e9, rabi_frequency=12.5e6),
                "adc": ReadoutConfig(target="q"),
            }
        ),
    )

Frequencies are in Hz and times in ns. ``rabi_frequency`` is the Rabi frequency
of a full-amplitude pulse, so a 40 ns rectangular pulse is a π pulse:

.. code-block:: python

    pulse = Pulse(duration=40, amplitude=1, envelope=Rectangular())
    acquisition = Acquisition(duration=100)
    sequence = PulseSequence([("xy", pulse)])
    sequence.align(["xy", "adc"])
    sequence.append(("adc", acquisition))

    platform.connect()
    try:
        result = platform.execute([sequence], averaging_mode=AveragingMode.CYCLIC)
    finally:
        platform.disconnect()
    print(result[acquisition.id])  # Approximately 1: the excited population.

Results are keyed by acquisition identifier. Averaged discrimination returns
the exact probability of leaving the ground state; single-shot modes sample
outcomes from it. See :doc:`../guides/experiments` for every acquisition mode.

Next, sweep the pulse amplitude in :doc:`../tutorials/rabi`.
