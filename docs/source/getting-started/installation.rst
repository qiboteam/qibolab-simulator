Installation
============

Python 3.11 or newer is required. The package is not yet on PyPI: install it
from the repository.

.. code-block:: sh

    uv add "qibolab-simulator @ git+https://github.com/qiboteam/qibolab-simulator"

or, with ``pip``:

.. code-block:: sh

    pip install "qibolab-simulator @ git+https://github.com/qiboteam/qibolab-simulator"

Optional extras
---------------

``dynamiqs``
    The batched :doc:`dynamiqs backend <../guides/solvers>`, on CPU, GPU, or
    TPU. Install JAX for your accelerator as described in
    `its documentation <https://docs.jax.dev/en/latest/installation.html>`_,
    e.g. ``pip install -U "jax[cuda12]"``.

``distributed``
    `Dask <https://distributed.dask.org>`_, to run sweeps on a
    :doc:`caller-managed cluster <../guides/parallel>`.

.. code-block:: sh

    pip install "qibolab-simulator[dynamiqs,distributed] @ git+https://github.com/qiboteam/qibolab-simulator"

From a checkout
---------------

The repository contains the examples used in these tutorials:

.. code-block:: sh

    git clone https://github.com/qiboteam/qibolab-simulator
    cd qibolab-simulator
    uv sync
    uv run python examples/rabi.py

The Qibocal :doc:`calibration notebook <../tutorials/calibration>` needs the
``notebooks`` dependency group (Python 3.11–3.13):

.. code-block:: sh

    uv sync --group notebooks
    uv run marimo edit notebooks/calibration.py
