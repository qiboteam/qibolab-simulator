Parallel execution
==================

Sweep points are independent experiments: the controller compiles them lazily,
groups them in batches, and solves the batches in the calling process, on
local worker processes, or on a caller-managed executor such as a Dask
cluster.

Start with one worker
---------------------

``SimulatorController()`` uses one worker, in the calling process. This is the
recommended baseline on a laptop: worker startup, serialization, and
scheduling may cost more than they save on a small calibration. Measure the
crossover on your experiments, with the end-to-end benchmark:

.. code-block:: sh

    uv run python benchmarks/rabi.py --points 201 --workers 1 2 4 --repeats 3

It reports first-call and warm timings, and the error against an analytic
reference.

Local processes
---------------

.. code-block:: python

    controller = SimulatorController(
        workers=4, batch_size=8, max_in_flight=8, blas_threads=1, seed=1234
    )

.. list-table::
    :header-rows: 1

    * - Field
      - Meaning
    * - ``workers``
      - Local processes; must stay ``1`` with an external executor
    * - ``batch_size``
      - Experiments per task; default ``8`` for QuTiP, ``1024`` for dynamiqs
    * - ``max_in_flight``
      - Submitted, unconsumed batches; default ``2 * workers``
    * - ``blas_threads``
      - Numerical library threads in each worker; default ``1``
    * - ``seed``
      - Optional seed for reproducible shots

Workers are started with ``spawn``: guard scripts with
``if __name__ == "__main__":``. ``platform.connect()`` starts the pool, and
``platform.disconnect()`` stops it, so repeated executions reuse it; without a
connection, each execution creates and releases its own.

Larger batches amortize scheduling, but balance uneven solve times worse.
Bounded submission limits the queued jobs, not the collected results: workers
return measurements, not states, but single-shot results can still be large.

Dask cluster
------------

With the ``distributed`` extra, pass a Dask executor, keeping ``workers=1``.
The caller owns the cluster: the controller never shuts it down. Configure
**process workers with one thread each**, e.g.
``LocalCluster(processes=True, threads_per_worker=1)``: QuTiP solvers are not
thread-safe, and the controller rejects thread pools.

.. code-block:: python

    import os

    from distributed import Client
    from qibolab import AveragingMode

    from examples.rabi import create_platform, rabi
    from qibolab_simulator import SimulatorController


    def main():
        worker_count = int(os.environ["SIMULATOR_WORKERS"])
        with Client(os.environ["DASK_SCHEDULER_ADDRESS"]) as client:
            with client.get_executor() as executor:
                platform = create_platform()
                channels = platform.instruments["simulator"].channels
                platform.instruments["simulator"] = SimulatorController(
                    channels=channels,
                    executor=executor,
                    workers=1,  # Local pool disabled; not the cluster size.
                    batch_size=8,
                    max_in_flight=2 * worker_count,
                    seed=1234,
                )
                sequence, acquisition, sweeper = rabi()
                platform.connect()
                try:
                    result = platform.execute(
                        [sequence], [[sweeper]], averaging_mode=AveragingMode.CYCLIC
                    )
                    print(result[acquisition.id])
                finally:
                    platform.disconnect()


    if __name__ == "__main__":
        main()

Set ``max_in_flight`` explicitly, to at least twice the number of cluster
workers: since ``workers=1``, its default only keeps two batches in flight.
Install the same package versions on every worker.

Accelerators
------------

The dynamiqs backend batches a whole sweep on one device: keep ``workers=1``
for a single accelerator. Several local or Dask workers each run their own JAX
process, which suits one worker per device.
