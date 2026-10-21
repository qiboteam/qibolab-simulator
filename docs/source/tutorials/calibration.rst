Calibration with Qibocal
========================

`Qibocal <https://github.com/qiboteam/qibocal>`_ protocols run on a simulated
platform exactly as on hardware. The `calibration notebook
<https://github.com/qiboteam/qibolab-simulator/blob/main/notebooks/calibration.py>`_,
a `marimo <https://marimo.io>`_ notebook, calibrates the
:doc:`two-qutrit platform <qutrits>`, displaying data, fits, and reports with
`qibocal-widgets <https://github.com/qiboteam/qibocal-widgets>`_.

.. only:: html

    An `executed copy of the notebook <../calibration.html>`_ is built with
    this documentation.

Running the notebook
--------------------

The notebook uses Qibocal's ``Executor`` and ``BoundProtocol`` interface,
currently developed on its ``protodev`` branch, which the ``notebooks``
dependency group tracks. From the repository root, with Python 3.11–3.13:

.. code-block:: sh

    uv sync --group notebooks
    uv run marimo edit notebooks/calibration.py

Configuration is code: edit the protocol calls, ``targets``, and ``nshots``
in their cells. Set ``coupled = False`` in the first cell for faster
single-qubit experiments.

Executor phases
---------------

The first section runs each phase of a Ramsey experiment separately, on both
qubits:

.. code-block:: python

    from qibocal import create_calibration_platform, protocols
    from qibocal_widgets.marimo import MarimoExecutor

    platform = create_calibration_platform("qutrits")
    executor = MarimoExecutor.create(platform=platform, path="var/report/", update=False)

    bound = protocols.ramsey(delay=(10, 1000, 100))
    with executor:
        acquired = executor.acquire(bound, nshots=1000, targets=[0, 1])
    fitted = executor.fit(acquired)
    reported = executor.report(fitted)
    executor.update(reported)  # Optional: applies the fitted parameters.

The second runs a whole operation, a flux scan of the first qubit's frequency,
with a single ``executor(bound)`` call. Every phase returns a ``Completed`` object,
which renders its data and reports as the cell output.

Both sections share the executor and its in-memory platform: updates apply to
later experiments, until the platform is created again. The example runcard is
never modified.

What to expect
--------------

Rabi amplitude and length, T1, T2, spin echo, qubit spectroscopy, and flux
scans all run on the simulator. Keep in mind that:

- Spectroscopy signals are driven non-ground populations, not resonator
  responses.
- Coupling shifts transition frequencies: run spectroscopy before Ramsey.
- Short coherence windows are cheap but give unreliable T1 and T2 fits; inspect
  the fit quality even when Qibocal reports success.
- Readout is binary, so there is no ternary discrimination, resonator
  calibration, or IQ-cloud classification.
