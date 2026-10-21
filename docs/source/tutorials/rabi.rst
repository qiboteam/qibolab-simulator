Rabi oscillations
=================

This tutorial sweeps the amplitude of a drive pulse on a single qubit, then
adds a frequency axis and single-shot sampling. The complete script is
`examples/rabi.py
<https://github.com/qiboteam/qibolab-simulator/blob/main/examples/rabi.py>`_.

Platform
--------

The platform is the one of the :doc:`../getting-started/quickstart`, built by
a function so that worker processes can recreate it:

.. literalinclude:: ../../../examples/rabi.py
    :pyobject: create_platform

Amplitude sweep
---------------

Sweepers are those of Qibolab. Timing and alignment are resolved again at each
sweep point, so duration and delay sweeps move later pulses and acquisitions:

.. literalinclude:: ../../../examples/rabi.py
    :pyobject: rabi

.. code-block:: python

    platform = create_platform()
    sequence, readout, sweeper = rabi()
    platform.connect()
    try:
        results = platform.execute(
            [sequence], [[sweeper]], averaging_mode=AveragingMode.CYCLIC
        )
    finally:
        platform.disconnect()
    results[readout.id]  # sin²(πA/2), for each amplitude A

Chevron
-------

Sweepers within one inner list vary together; separate lists form the axes of
a Cartesian product. Adding a drive frequency axis gives a chevron, of shape
``(21, 41)``:

.. code-block:: python

    from qibolab import Parameter, Sweeper

    frequency = Sweeper(
        parameter=Parameter.frequency,
        channels=["drive"],
        values=5e9 + np.linspace(-20e6, 20e6, 41),
    )
    results = platform.execute(
        [sequence], [[sweeper], [frequency]], averaging_mode=AveragingMode.CYCLIC
    )

Single shots
------------

With ``AveragingMode.SINGLESHOT``, each shot is a binary outcome, drawn from
the simulated populations, and results gain a leading shot axis. The
controller's ``seed`` makes the draws reproducible:

.. code-block:: python

    results = platform.execute(
        [sequence], [[sweeper]], nshots=1000, averaging_mode=AveragingMode.SINGLESHOT
    )
    results[readout.id].shape  # (1000, 21)

Exact propagation
-----------------

By default, the QuTiP backend integrates the dynamics adaptively. Evolving each
pulse sample with one matrix exponential is usually faster and more accurate:

.. code-block:: python

    from qibolab_simulator import Settings

    platform.instruments["simulator"].solver = Settings(propagation="exact")

See :doc:`../guides/solvers` for the trade-offs.
