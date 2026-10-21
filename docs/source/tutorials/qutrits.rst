Coupled qutrits
===============

The repository contains a two-qutrit platform, converted from Qibolab's
split-transmon emulator platform: two three-level transmons with a 12 MHz
exchange coupling, T1 and T2, flux channels, and calibrated Gaussian RX, RX90,
and RX12 natives, as well as a flux-pulse CZ. This tutorial runs a Ramsey
experiment on it.

Loading the platform
--------------------

From the repository root, either import it directly:

.. code-block:: python

    from examples.platforms.qutrits import create_platform

    platform = create_platform()

or point Qibolab's loader to the examples:

.. code-block:: sh

    export QIBOLAB_PLATFORMS="$PWD/examples/platforms"

.. code-block:: python

    from qibolab import create_platform

    platform = create_platform("qutrits")

Its runcard, `parameters.json
<https://github.com/qiboteam/qibolab-simulator/blob/main/examples/platforms/qutrits/parameters.json>`_,
is a complete example of the ``simulator-*`` configuration kinds. The flux
channels' offsets bias each transmon at its sweetspot.

Ramsey
------

Two RX90 natives, separated by a swept delay, with the drive detuned by 2 MHz
to produce fringes. The single-point frequency sweeper sets the detuning
without editing the runcard:

.. code-block:: python

    import numpy as np
    from qibolab import AveragingMode, Delay, Parameter, PulseSequence, Sweeper

    natives = platform.natives.single_qubit[0]
    delay = Delay(duration=0)
    sequence = PulseSequence()
    sequence += natives.RX90()
    sequence.append(("0/drive", delay))
    sequence += natives.RX90()
    sequence.align(["0/drive", "0/acquisition"])
    sequence += natives.MZ()
    acquisition = sequence.acquisitions[0][1]

    delays = Sweeper(
        parameter=Parameter.duration, pulses=[delay], values=np.arange(0, 2000, 20.0)
    )
    frequency = Sweeper(
        parameter=Parameter.frequency,
        channels=["0/drive"],
        values=np.array([platform.config("0/drive").frequency + 2e6]),
    )
    platform.connect()
    try:
        results = platform.execute(
            [sequence], [[delays], [frequency]], averaging_mode=AveragingMode.CYCLIC
        )
    finally:
        platform.disconnect()
    results[acquisition.id][:, 0]  # Damped fringes, one per sweep point

The second qubit stays in its ground state, but its coupling still shifts the
first qubit's frequency, and both are part of the simulation: with three levels
each, the density matrix is 9 × 9.

Faster sweeps
-------------

This sweep takes about 15 s with the default adaptive solver on a laptop CPU.
Exact propagation, which exponentiates each pulse sample, runs it in about
1.5 s:

.. code-block:: python

    from qibolab_simulator import Settings

    platform.instruments["simulator"].solver = Settings(propagation="exact")

Independent sweep points can also run on several processes, or as a single
batch on a GPU. See :doc:`../guides/solvers` and :doc:`../guides/parallel`.

Modeling choices
----------------

The conversion from the original emulator platform changes its physical and
readout contract:

- Fundamental-transition T1 and T2 are scalar oscillator parameters. The
  1–2 transition decays as a damped oscillator, with a T1/2 lifetime of the
  second excited level.
- Readout is ideal: both excited levels count as non-ground.
- The RX12 and CZ parameters are the upstream starting points, not gates
  recalibrated for this model.
- Each shot starts in the ground state: relaxation time does not simulate
  memory between shots.
