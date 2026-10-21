Migrating from the Qibolab emulator
===================================

The ``simulator-*`` configuration kinds are **not** a drop-in replacement for
the runcards of the emulator distributed with Qibolab 0.2. Port platforms
deliberately:

- Import from ``qibolab_simulator`` and call
  :func:`~qibolab_simulator.register_configs` before loading runcards, to
  register ``simulator-hamiltonian``, ``simulator-drive``, ``simulator-flux``,
  and ``simulator-readout``.
- Put the model under the ``"hamiltonian"`` configuration key, and give every
  simulator channel an explicit ``target``.
- Move level counts to each :class:`~qibolab_simulator.QubitConfig`.
- Convert per-transition T1 and T2 to scalar parameters **only if** the
  intended physics is fundamental-transition damping of an oscillator.
  Otherwise, this model is not an equivalent replacement.

The :doc:`two-qutrit platform <../tutorials/qutrits>` is such a conversion,
with its modeling changes listed.

Relation to the Qibolab roadmap
-------------------------------

The upstream roadmap motivates this package, without promising feature parity:

- `Independent packaging (#1257) <https://github.com/qiboteam/qibolab/issues/1257>`_
  and `rotating frames (#1453) <https://github.com/qiboteam/qibolab/issues/1453>`_
  are the main direction of this driver.
- `Collapsed measurements (#1263) <https://github.com/qiboteam/qibolab/issues/1263>`_:
  only terminal measurements are supported.
- `RAW traces (#1173) <https://github.com/qiboteam/qibolab/issues/1173>`_ and
  `confusion matrices (#1413) <https://github.com/qiboteam/qibolab/issues/1413>`_:
  there is no readout-device or assignment-error model.
- `External operators (#1256) <https://github.com/qiboteam/qibolab/issues/1256>`_
  and `diagnostics (#1409) <https://github.com/qiboteam/qibolab/issues/1409>`_:
  arbitrary operators and Hamiltonian or state traces are not provided.
