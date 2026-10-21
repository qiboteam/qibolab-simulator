Experiments
===========

Sequences
---------

Sequences are ordinary Qibolab pulse sequences, with ``Pulse``, ``Delay``,
``VirtualZ``, ``Align``, ``Acquisition``, and ``Readout`` instructions on
drive, flux, and acquisition channels. Their timing is resolved for each sweep
point, after the sweep's updates.

The simulator rejects, rather than approximates:

- acquisitions that are not simultaneous, or followed by control pulses
  (mid-circuit measurements);
- RAW acquisition, active reset, and chirped pulses;
- nonzero readout ``delay``, ``smearing``, and ``time_of_flight``;
- control pulses off the sampling grid.

Acquisitions
------------

Each acquisition reports whether its target left the ground state.

.. list-table::
    :header-rows: 1

    * - Mode
      - Result per acquisition
    * - Averaged discrimination
      - Exact non-ground probability
    * - Single-shot discrimination
      - Binary non-ground outcome per shot
    * - Averaged integration
      - Ideal IQ pair ``(probability, 0)``
    * - Single-shot integration
      - Ideal IQ pair ``(outcome, 0)`` per shot

Single shots draw the **joint** state once per shot, so simultaneous
acquisitions retain their correlations. There is no assignment error, IQ
noise, resonator response, or ternary discrimination.

For a sweep of shape ``S``, discrimination results have shape ``S`` when
averaged and ``(nshots, *S)`` for single shots. Integration adds a last axis
of length two. Acquisition identifiers must be unique across the sequences of
one execution.

Sweeps
------

Sweepers are passed as ``[[a, b], [c]]``: sweepers in one inner list vary
together, and must have the same length; the outer list forms Cartesian axes.
Supported parameters:

- pulse ``amplitude``, ``relative_phase``, ``duration``, and
  ``duration_interpolated``;
- delay ``duration``;
- drive channel ``frequency``;
- flux channel ``offset``.

Pulse durations stay on the sampling grid: ``duration_interpolated`` resamples
the envelope on that grid, and does not emulate an instrument's interpolation.

Reproducibility
---------------

With a ``seed``, the controller draws shots from streams determined by
sequence index and sweep point, not by worker, batch, or completion order.
Repeated executions therefore return the same shots, locally or on a cluster.
Change the seed for independent repeats. Different solvers or machines may
still produce slightly different probabilities, and thus occasionally
different shots.
