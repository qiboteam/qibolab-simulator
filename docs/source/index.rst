Qibolab simulator
=================

A `Qibolab <https://github.com/qiboteam/qibolab>`_ driver for pulse-level
simulation, behind the same ``Platform.execute()`` interface as real hardware.
It targets calibration experiments on small systems: multilevel transmons
(Duffing oscillators) with exchange couplings, drive and flux channels, and T1
and T2, evolved in a rotating frame with `QuTiP <https://qutip.org>`_ or,
batched on GPU, with `dynamiqs <https://www.dynamiqs.org>`_.

.. code-block:: python

    from qibolab import AveragingMode

    platform.connect()
    results = platform.execute([sequence], [[sweeper]], averaging_mode=AveragingMode.CYCLIC)
    platform.disconnect()

- Platforms are ordinary Qibolab platforms: only the controller and the
  ``simulator-*`` channel configurations are specific to the simulator.
- Sweeps resolve pulse timing and alignment at each point, as hardware does.
- Acquisitions return ideal discriminated or integrated results, sampled from
  the joint final state, so simultaneous readouts stay correlated.
- Sweeps can run on local processes, on a Dask cluster, or as JAX batches.

.. warning::

    The model uses the rotating-wave approximation, and resets to the ground
    state before each experiment. Readout is ideal: there is no resonator,
    RAW trace, mid-circuit measurement, or active reset. Unsupported features
    fail explicitly. See :doc:`guides/model`.

.. toctree::
    :maxdepth: 2
    :caption: Getting started

    getting-started/installation
    getting-started/quickstart

.. toctree::
    :maxdepth: 2
    :caption: Tutorials

    tutorials/rabi
    tutorials/qutrits
    tutorials/calibration

.. toctree::
    :maxdepth: 2
    :caption: Guides

    guides/model
    guides/experiments
    guides/solvers
    guides/parallel
    guides/migration

.. toctree::
    :maxdepth: 2
    :caption: Reference

    reference/api
    reference/development

.. toctree::
    :maxdepth: 1
    :caption: Qibo ecosystem

    Qibo <https://qibo.science/qibo/stable/>
    Qibolab <https://qibo.science/qibolab/stable/>
    Qibocal <https://qibo.science/qibocal/stable/>


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
