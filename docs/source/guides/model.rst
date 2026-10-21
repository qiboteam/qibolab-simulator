Physical model
==============

Units
-----

- Frequencies, anharmonicities, exchange couplings, and ``rabi_frequency``:
  **Hz**.
- Durations, ``t1``, and ``t2``: **ns**. Phases: **radians**.
- ``sampling_rate_`` of the controller: samples per ns (default ``1``). Pulse
  durations must lie on this sampling grid.
- Flux offsets and flux-pulse amplitudes are converted to flux quanta with
  ``voltage_to_flux``; ``sweetspot`` is in flux quanta.

Hamiltonian
-----------

Each subsystem is a Duffing oscillator truncated to its own number of
``levels``, with frequency :math:`\omega_i` and anharmonicity :math:`\alpha_i`.
Couplings are excitation-conserving exchange terms, of strength :math:`g_{ij}`:

.. math::

    H = \sum_i \left[\omega_i n_i + \frac{\alpha_i}{2} n_i (n_i - 1)\right]
      + \sum_{(i, j)} g_{ij} \left(a_i^\dagger a_j + a_i a_j^\dagger\right)

Each exchange-connected component evolves in one rotating frame, at the
frequency of its first qubit, in the configured order. This removes a common
carrier, but not the detunings between coupled qubits.

Drives
------

A drive channel acts on the creation operator of its target with coefficient

.. math::

    -i \pi \Omega \, (I + iQ) \, e^{-i [2\pi (f_d - f_r) t + \phi]}

where :math:`\Omega` is ``rabi_frequency``, :math:`I + iQ` the amplitude-scaled
envelope samples, :math:`f_d` the drive frequency, :math:`f_r` the frame
frequency, and :math:`\phi` the pulse's ``relative_phase`` plus the channel's
accumulated ``VirtualZ`` phases. The hermitian conjugate acts on the
annihilation operator. Envelopes are sampled once per sampling period and held
constant in between, as by a digital-to-analog converter.

.. warning::

    The rotating-wave approximation drops counter-rotating terms. This is not an
    exact change of coordinates of the lab-frame driven model: validate strong
    drives and large detunings before relying on it.

Flux
----

Flux channels use the split-transmon frequency-versus-flux relation, with
``sweetspot`` and junction ``asymmetry``:

.. math::

    \omega(\Phi) = (\omega - \alpha) \left[d^2 + (1 - d^2)
        \cos^2\big(\pi (\Phi - \Phi_0)\big)\right]^{1/4} + \alpha

The channel ``offset`` biases the qubit statically; flux pulses add to it, and
must have real envelopes, without a phase. Only one flux channel per subsystem
is supported.

Decoherence
-----------

``t1`` and ``t2`` are optional, scalar, fundamental-transition times, with
``t2 <= 2 * t1``. They become Lindblad dissipators: relaxation acts on the
annihilation operator, pure dephasing on the number operator, so higher levels
decay and dephase as those of an oscillator. Independent rates for each
transition cannot be configured.

Initial state and readout
-------------------------

Every experiment starts in the joint ground state, and is measured once, at
its end. Readout is ideal and binary: every level above the ground state reads
as ``1``. See :doc:`experiments`.
