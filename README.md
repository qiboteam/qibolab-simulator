# Qibolab simulator

A standalone QuTiP driver for pulse-level simulation through Qibolab's execution
interface. It targets small-system calibration experiments, with rotating-frame
evolution, mixed-level Duffing oscillators, and bounded parallel sweep execution.

## Install and run

Python 3.11 or newer is required. From this checkout:

```sh
uv sync
uv run python examples/rabi.py
uv run python benchmarks/rabi.py --workers 1 2 --points 201
```

Alternatively, install into your environment with `pip install -e .` and run
the scripts with `python`. For a caller-managed Dask cluster, use
`uv sync --extra distributed` or `pip install -e '.[distributed]'`.

With [devenv](https://devenv.sh/), run `devenv shell` to enter a Python 3.12
environment with development, notebook, and distributed dependencies installed.
It sets `QIBOLAB_PLATFORMS` to this checkout's `examples/platforms`, so Qibolab
can load the `split_transmons` platform by name.

## Scope

- Configurable subsystem levels, exchange coupling, drive and flux channels,
  and scalar fundamental-transition T1/T2 parameters.
- Pulse and channel sweeps, with timing and alignment resolved at each point.
- Simultaneous **terminal** ideal acquisitions: correlated joint shots or exact
  averaged non-ground populations; synthetic integration returns `(value, 0)`.
- One worker by default; reusable local process pools or an externally supplied
  process-based executor, including Dask.

This is an **RWA model**, not an exact full lab-frame emulator. It resets to the
ground state for each experiment and does not model RAW readout, mid-circuit
measurement, active reset, chirped pulses, or readout noise. Unsupported operations
fail explicitly rather than silently approximating those features.

The new `simulator-*` configuration kinds are **not a drop-in replacement for old
emulator runcards**. See [usage, model, and migration](docs/usage.md) before porting
a platform. See [execution and scaling](docs/scaling.md) for batching, reproducible
sampling, cluster use, and the limits of exact Hilbert-space simulation.

There is no universal speedup claim: benchmark your experiment, including worker
startup, serialization, and numerical accuracy. A single worker is the recommended
starting point on a laptop.

## Development and CI

Run the CI tasks locally with `uv run --with poethepoet poe lint` and
`uv run --with poethepoet poe test`. Install `--extra distributed` to include
the Dask tests, or `--group notebooks` to include the Qibocal integration tests.

CI has separate tests, docs, and wheels workflows using the shared
[qiboteam workflows](https://github.com/qiboteam/workflows) on the `uv` ref,
like Qibolab and Qibocal. The docs workflow exports the calibration notebook
as an HTML artifact; wheel publishing is disabled. Coverage uploads require
the `codecov_token` GitHub Actions secret while this repository is private.

## Examples and license

- [Minimal platform and Rabi sweep](examples/rabi.py)
- [End-to-end Rabi benchmark](benchmarks/rabi.py)
- [Two coupled split-transmon qutrits](examples/platforms/split_transmons/README.md),
  with RX/RX90/RX12 and a flux-driven CZ pulse.
- [Code-driven Qibocal calibration notebook](notebooks/calibration.py):
  `uv sync --group notebooks`, then
  `uv run --group notebooks marimo edit notebooks/calibration.py`
  (Python 3.11-3.13).

Licensed under Apache-2.0; see [LICENSE](LICENSE).
