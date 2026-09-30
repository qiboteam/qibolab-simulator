# Execution and scaling

## Start with one worker

`SimulatorController()` defaults to `workers=1`: execution stays in the calling
process. This is the recommended laptop baseline. Rotating frames can remove
carrier-scale integration work, but neither multiprocessing nor a cluster is
guaranteed to accelerate a small calibration.

```sh
uv run python benchmarks/rabi.py --points 201 --workers 1 2 4 --repeats 3
```

The benchmark reports first-call and warm timings plus error against an analytic
Rabi reference. Compare identical physics and accuracy settings; the RWA is not
an exact full lab-frame model. Measure the crossover where useful solve time
exceeds startup, compilation, serialization, and scheduling costs.

Closed-system intervals with a constant diagonal Hamiltonian use exact
basis-state phases instead of an adaptive solver. This avoids integration work
over long idle delays, including anharmonic multilevel phases and constant flux
shifts, while retaining the phases needed by later pulses. It requires only the
Hamiltonian diagonal and state vector, not a dense propagator. Driven,
non-diagonal, time-dependent, and dissipative intervals still use QuTiP.

## Local process execution

```python
controller = SimulatorController(
    workers=4,
    batch_size=8,
    max_in_flight=8,
    blas_threads=1,
    seed=1234,
)
```

Use this configuration when constructing the controller in your platform.
Local workers use multiprocessing **spawn**; launch scripts under an
`if __name__ == "__main__":` guard. Call `platform.connect()` before repeated
executions and `platform.disconnect()` in a `finally` block. This retains and
reuses the process pool across calls rather than paying startup each time.
Without an explicit connection, a multiworker `play()` creates and releases
its own pool for that call.

| Control | Meaning |
| --- | --- |
| `workers` | Local process count; must remain `1` with an external executor |
| `batch_size` | Experiments per submitted task; default `8` |
| `max_in_flight` | Maximum submitted, not-yet-consumed batches; defaults to `2 * workers` |
| `blas_threads` | Numerical-library thread limit inside each worker; default `1` |
| `seed` | Optional reproducible measurement-sampling seed |

Larger batches amortize scheduler overhead but worsen load balance for uneven
solve times. Smaller batches improve responsiveness at higher scheduling cost.
Bounded submission limits queued job data; it does **not** bound total collected
result size. Workers return acquisition measurements, not density-matrix traces.
Single-shot outputs can still be large.

Seed streams depend on sequence index and sweep coordinate, not worker identity,
batch size, or completion order. With a fixed seed, repeated plays repeat the
measurement stream, including across local/distributed scheduling. Change the
seed for independent repeats. This is a sampling guarantee, not a promise of
bitwise solver agreement across different numerical libraries or machines.

## Caller-managed Dask cluster

Install `.[distributed]` (or use `uv sync --extra distributed`). Start and manage
the cluster outside this package. Configure **process workers with one execution
thread each**; shared-thread QuTiP solves are not supported. The controller
rejects a standard `ThreadPoolExecutor`, but the caller must ensure a distributed
executor actually uses the required worker layout.

For a local Dask deployment, use `LocalCluster(processes=True,
threads_per_worker=1, n_workers=...)`. On Slurm, provision Dask workers with your
site's existing job scripts or Dask jobqueue setup; this driver needs no new
scheduler integration. Install compatible package versions on every worker.

Example from the repository root, connecting to an existing scheduler whose
address and worker count are supplied through the environment:

```python
import os

from distributed import Client
from examples.rabi import create_platform, rabi
from qibolab import AveragingMode
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
                blas_threads=1,
                seed=1234,
            )
            sequence, acquisition, sweeper = rabi()
            platform.connect()
            try:
                result = platform.execute(
                    [sequence],
                    [[sweeper]],
                    averaging_mode=AveragingMode.CYCLIC,
                )
                print(result[acquisition.id])
            finally:
                platform.disconnect()


if __name__ == "__main__":
    main()
```

Set `max_in_flight` explicitly for the cluster—start with at least
`2 * worker_count`, then tune memory and load balance. Because external executors
require `workers=1`, the implicit default is only **two batches**, regardless of
cluster size. The caller owns the executor/client/cluster and any credentials or
TLS configuration. `disconnect()` never shuts down an external executor.

## What does not scale away

For subsystem dimensions `d_i`, the exact Hilbert dimension is `D = product(d_i)`.
A state vector needs `O(D)` storage; a density matrix needs `O(D²)`. Operators,
solver workspace, and possible Liouvillian representations add further cost.
Adding levels, couplers, or spectator subsystems can dominate any frame or
parallelization improvement.

This implementation distributes **independent experiments**, not one large
Hilbert-space solve across machines. Each worker must fit its own solve in
memory; more workers multiply concurrent memory demand. Ground initialization
and terminal-only acquisitions make these jobs independent, but do not
approximate away subsystem correlations within a solve.

Use one numerical thread per worker initially to avoid oversubscription, then
benchmark deliberately. Increase system size, sweep size, and worker count
separately, checking accuracy as well as throughput. This is CPU/QuTiP execution;
an external Dask executor does not by itself enable GPU solving.
