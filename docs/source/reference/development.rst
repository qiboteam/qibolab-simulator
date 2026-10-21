Development
===========

Setup
-----

Install `uv <https://docs.astral.sh/uv/>`_, then, from a checkout:

.. code-block:: sh

    uv sync --locked --extra distributed --extra dynamiqs
    uv run --with poethepoet poe test
    uv run --with poethepoet poe lint

`devenv <https://devenv.sh/>`_ users can run ``devenv shell`` instead: it also
installs the notebook dependencies, and sets ``QIBOLAB_PLATFORMS`` to the
example platforms.

``uv.lock`` records the development environment: run ``uv lock`` after changing
dependencies, and commit it. The Dask, dynamiqs, and Qibocal tests skip when
their optional dependencies are missing.

Documentation
-------------

The documentation is built with Sphinx, together with an executed export of
the calibration notebook:

.. code-block:: sh

    uv sync --locked --group docs --group notebooks --extra dynamiqs
    uv run --with poethepoet poe docs

The site is written to ``docs/build/html``. ``poe sphinx`` rebuilds the pages
alone, keeping any previous notebook export.

Architecture
------------

The package has four layers:

#. **Configuration and compilation** (``config``, ``sequence``): serializable
   runcard kinds, and the compilation of a swept sequence into a plain,
   pickleable :class:`~qibolab_simulator.sequence.Experiment`.
#. **Engines** (``engine``, ``dynamiqs_engine``): Qibolab-independent
   solvers. They share the model, frames, operators, segmentation, and
   validation, and must agree up to solver accuracy.
#. **Scheduling** (``sweep``, ``execution``): lazy expansion of sweeps, and
   bounded submission of batches to processes or executors.
#. **Measurement** (``controller``): sampling of acquisitions from the final
   populations, with streams tied to sweep coordinates.

Keep solver jobs pickleable, random streams tied to experiment coordinates, and
unsupported physics explicit. A new physical approximation needs an accuracy
regression, not just a smoke test: ``tests/test_engine.py`` checks analytic
dynamics, ``tests/test_exact.py`` exact propagation against strict references,
and ``tests/test_dynamiqs.py`` every physical case of the dynamiqs backend
against QuTiP. Pass coefficient arrays to JAX as traced arguments, never through
closures, so that compiled solvers are reused.

Qibolab platforms replace the process's termination handlers. Notebooks restore
them after creating a platform, since marimo needs them to stop its kernel.

Benchmarks
----------

.. code-block:: sh

    uv run python benchmarks/rabi.py --points 201 --workers 1 2 --repeats 3
    uv run --with qibo python benchmarks/rabi.py --points 21 --workers 1 --legacy

The benchmark covers compilation, scheduling, integration, and result
conversion, with first-call and warm timings and the error against an analytic
reference. ``--legacy`` compares with the emulator of the installed Qibolab
release, with the same negligible damping in both engines.

Continuous integration
----------------------

All workflows reuse `qiboteam/workflows <https://github.com/qiboteam/workflows>`_.

- **Tests**: Linux on Python 3.11 and 3.13, and macOS on Python 3.12, including
  process spawning, Dask, and dynamiqs. Coverage uploads use the
  ``codecov_token`` secret.
- **Build wheels**: wheel and sdist, without publishing.
- **Docs**: builds the documentation on pull requests, and deploys it to GitHub
  Pages from ``main`` (as ``latest``) and from tags (as ``stable``).
