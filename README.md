# Qibolab simulator

A [Qibolab](https://github.com/qiboteam/qibolab) driver for pulse-level
simulation, behind the same `Platform.execute()` interface as real hardware.
It targets calibration experiments on small systems: multilevel transmons with
exchange couplings, drive and flux channels, and T1 and T2, evolved in a
rotating frame with QuTiP or, batched on GPU, with dynamiqs.

📖 **[Documentation](https://qibo.science/qibolab-simulator/latest/)**:
installation, tutorials, physical model, solvers, parallel execution, and API
reference.

## Install and run

Python 3.11 or newer is required. From this checkout:

```sh
uv sync
uv run python examples/rabi.py
```

Optional extras: `dynamiqs` for the batched backend, on CPU, GPU, or TPU, and
`distributed` for Dask clusters.

## Scope

- Configurable subsystem levels, exchange couplings, drive and flux channels,
  and scalar fundamental-transition T1/T2.
- Pulse and channel sweeps, with timing and alignment resolved at each point.
- Simultaneous terminal ideal acquisitions: correlated joint shots or exact
  averaged non-ground populations.
- Exact piecewise-constant propagation, local process pools, caller-managed
  executors such as Dask, and batched GPU solving.

This is a **rotating-wave model**, not an exact lab-frame emulator. It resets
to the ground state for each experiment, and does not model RAW readout,
mid-circuit measurement, active reset, chirped pulses, or readout noise:
unsupported operations fail explicitly. Its `simulator-*` configuration kinds
are not a drop-in replacement for the runcards of Qibolab's emulator.

## Examples

- [Minimal platform and Rabi sweep](examples/rabi.py)
- [Two coupled qutrits](examples/platforms/qutrits/README.md), with
  RX/RX90/RX12 and a flux-driven CZ pulse
- [Qibocal calibration notebook](notebooks/calibration.py) and
  [QuTiP and dynamiqs performance notebook](notebooks/performance.py), with
  `uv sync --group notebooks`

## Development

```sh
uv run --with poethepoet poe test
uv run --with poethepoet poe lint
uv run --with poethepoet poe docs
```

See the [development guide](https://qibo.science/qibolab-simulator/latest/reference/development.html).
Licensed under Apache-2.0; see [LICENSE](LICENSE).
