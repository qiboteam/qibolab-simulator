Solvers
=======

The controller's ``backend`` selects QuTiP (the default) or dynamiqs. Both
share the model, pulse compilation, rotating frames, tolerances, and seeded
sampling, and agree up to solver accuracy.

.. code-block:: python

    from qibolab_simulator import DynamiqsSettings, Settings, SimulatorController

    controller = SimulatorController(
        backend="qutip",
        solver=Settings(rtol=1e-7, atol=1e-9, propagation="exact"),
        dynamiqs=DynamiqsSettings(method="dopri8", precision="double"),
    )

Every pulse boundary splits an experiment into segments, each solved from the
final state of the previous one. Closed systems evolve a state vector, of
dimension :math:`D = \prod_i d_i` for subsystem levels :math:`d_i`; open ones a
:math:`D \times D` density matrix. Both backends evolve constant, closed,
diagonal segments, such as idle qubits, with exact phases.

QuTiP, exact
------------

Pulse samples are constant between sample edges, so by default each sample
interval is evolved with one matrix exponential, of the Hamiltonian or of the
Lindbladian. This is exact for the simulated model and needs no tolerance. A
constant interval of any length, such as a delay or a rectangular pulse, costs
one exponential, with or without decoherence, and identical intervals, as in
symmetric envelopes or repeated delays, reuse theirs.

Detuned drives stay exact: during each segment, the rotating frame of every
coupled component follows its drives' detuning. When a component has
simultaneous drives at different frequencies, as in parallel calibrations of
coupled qubits, that segment falls back to the adaptive solver below.

Up to a generator dimension of 256 (:math:`D` for closed systems, :math:`D^2`
for open ones), exponentials are dense and cached; beyond it, their action on
the state is computed with sparse operators.

On the two-qutrit platform, with decoherence, exact propagation ran 64-point
detuned Rabi and Ramsey sweeps 4 to 9 times faster than the adaptive solver,
within 1e-9 of a strict reference, while the default tolerances deviated by up
to 2e-6.

QuTiP, adaptive
---------------

With ``propagation="adaptive"``, every segment is integrated with QuTiP's
adaptive ``dop853`` method, at the ``rtol`` and ``atol`` of
:class:`~qibolab_simulator.Settings`. Shaped pulses limit the step to half a
sample.

.. code-block:: python

    controller = SimulatorController(solver=Settings(propagation="adaptive"))

dynamiqs, batched
-----------------

With ``backend="dynamiqs"``, batches of experiments (``batch_size``, default
``1024``) are solved together by `dynamiqs <https://www.dynamiqs.org>`_, on
CPU, GPU, or TPU. It requires the ``dynamiqs`` extra, and JAX for the
accelerator.

- Experiments with the same structure (dimensions, decoherence, and shaped
  pulses with their sample grids) are solved together. Segments are rescaled to
  unit duration, so amplitude, phase, frequency, delay, and flux-offset sweeps
  share one structure, while each distinct pulse duration forms its own.
- Sweep order is free in simulation: the controller gathers points with the
  same structure across all sweep axes, up to eight batches at a time, before
  batching them. A chevron therefore fills whole batches for each duration,
  whatever the order of its axes. Results return to their sweep coordinates.
- Segments without time-dependent terms are exponentiated exactly; the others
  use an adaptive Runge-Kutta method that steps onto every sample edge.
- Batch, sample, and boundary counts are padded to powers of two. The first
  call compiles the solvers; later calls with the same shapes reuse them.

.. list-table:: :class:`~qibolab_simulator.DynamiqsSettings`
    :header-rows: 1

    * - Field
      - Values
    * - ``method``
      - ``"dopri8"`` (default, like ``dop853``), ``"dopri5"``, or ``"tsit5"``
    * - ``precision``
      - ``"double"`` (default) or ``"single"``, set process-wide in JAX
    * - ``device``
      - ``"cpu"``, ``"gpu"``, ``"tpu"``, or ``None`` for JAX's default
    * - ``frame``
      - ``"fixed"`` (default) or ``"drives"``

With ``frame="drives"``, frames follow the drives' detuning as in exact QuTiP
propagation, so detuned constant pulses, as in spectroscopies, are
exponentiated too. On a GPU, a 1024-point spectroscopy of the two-qutrit
platform ran 14 times faster, and about 1000 times more accurately. Shaped
pulses still need Runge-Kutta.

Single precision is much faster on most GPUs, but cannot meet the default
tolerances: loosen them, e.g. to ``Settings(rtol=1e-5, atol=1e-6)``. Between
segments, density matrices are made Hermitian again, and in single precision
their trace (or a state's norm) is restored, against rounding.

Choosing
--------

- Start with QuTiP, whose default exact propagation suits most experiments.
- On a multicore machine, use process workers: QuTiP sweeps scale almost
  linearly with them (see :doc:`parallel`).
- For large sweeps, try dynamiqs on a GPU; enable ``frame="drives"`` for
  spectroscopies. Include the first-call compilation in comparisons.
- Simulation cost grows with :math:`D`, and with :math:`D^2` for open
  systems: levels, couplers, and spectator qubits can outweigh any solver or
  parallelization gain.

The `performance notebook
<https://github.com/qiboteam/qibolab-simulator/blob/main/notebooks/performance.py>`_
compares the backends on the two-qutrit platform:

.. code-block:: sh

    uv run --group notebooks --extra dynamiqs marimo edit notebooks/performance.py
