API reference
=============

Platforms only need the package's top-level names: the controller, its
settings, and the runcard configurations. The engines below them can also be
used without Qibolab.

Controller
----------

.. autoclass:: qibolab_simulator.SimulatorController

.. autoclass:: qibolab_simulator.Settings

.. autoclass:: qibolab_simulator.DynamiqsSettings

Configurations
--------------

Runcard kinds, registered with Qibolab by :func:`~qibolab_simulator.register_configs`.

.. autofunction:: qibolab_simulator.register_configs

.. autoclass:: qibolab_simulator.HamiltonianConfig

.. autoclass:: qibolab_simulator.QubitConfig

.. autoclass:: qibolab_simulator.CouplingConfig

.. autoclass:: qibolab_simulator.DriveConfig

.. autoclass:: qibolab_simulator.FluxConfig

.. autoclass:: qibolab_simulator.ReadoutConfig

Engines
-------

Experiments compile to plain data, solved by Qibolab-independent engines, in
the units of :doc:`../guides/model`, with angular frequencies in rad/ns for
pulse samples.

.. autofunction:: qibolab_simulator.sequence.compile_sequence

.. autoclass:: qibolab_simulator.sequence.Experiment
    :no-members:

.. autoclass:: qibolab_simulator.engine.Model
    :no-members:

.. autoclass:: qibolab_simulator.engine.Qubit
    :no-members:

.. autoclass:: qibolab_simulator.engine.Coupling
    :no-members:

.. autoclass:: qibolab_simulator.engine.Drive
    :no-members:

.. autoclass:: qibolab_simulator.engine.Flux
    :no-members:

.. autofunction:: qibolab_simulator.engine.evolve

.. autofunction:: qibolab_simulator.dynamiqs_engine.evolve_batch

.. autofunction:: qibolab_simulator.engine.frame_frequencies
