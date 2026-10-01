# Development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --locked --extra distributed
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

`uv.lock` records the development environment. Use `uv lock` after changing
dependencies, and commit the updated lockfile. The package uses a `src` layout,
Hatchling for packaging, a `dev` dependency group, and an optional `notebooks`
group for Qibocal/marimo. Poe tasks provide the shared CI interface:
`uv run --with poethepoet poe lint` and `uv run --with poethepoet poe test`.

Tests cover analytic dynamics, pulse timing and sweeps, Qibolab `Platform.execute`
and runcard serialization, correlated readout, spawned workers, and a two-process
Dask cluster. The Dask test skips when the optional `distributed` extra is absent.
Run a focused module while iterating, for example `uv run pytest tests/test_sequence.py`.

## Qibocal and marimo

The [two-qutrit example](../examples/platforms/split_transmons/README.md) and
[calibration notebook](../notebooks/calibration.py) use Qibocal's `Executor` and
`BoundProtocol` from
[qiboteam/qibocal#1716](https://github.com/qiboteam/qibocal/pull/1716).
The optional group pins commit `cdd2133f0aa44db150bd935f22a87e0da40898c5`
of that unmerged PR. Python 3.11-3.13 is required for this optional group:

```sh
uv sync --locked --group notebooks
uv run --group notebooks marimo edit notebooks/calibration.py
uv run --group notebooks pytest tests/test_qibocal.py tests/test_split_transmons.py
uv run --group notebooks marimo check --strict notebooks/calibration.py
uv run --group notebooks marimo export html notebooks/calibration.py \
    -o /tmp/calibration.html -- --protocol rabi_amplitude
```

The notebook supports Rabi amplitude/length, detuned Ramsey, T1, T2, spin echo,
qubit spectroscopy, and flux-amplitude frequency calibration. It displays raw
data, Qibocal's fitted results and Plotly reports, and applies updates only on
request. Configuration is code-only: edit `coupled`, `targets`, `nshots`, the
`routine` and `parameters` dictionary, and `error_floor` in their cells.
The first section calls `executor.acquire()`, `executor.fit()`, and
`executor.report()` separately. The second calls `full_executor(bound)` once,
running the entire operation and presenting its `Completed` data, results, and
reports. The sections share protocol settings but use separate platforms so their
updates do not interfere. There are no buttons or input widgets; marimo UI is
used only for results presentation, and explanatory Markdown has its own cells.
Toggles live beside the operations they control: `coupled` in platform creation,
`error_floor` in fitting, and `apply_update` in the optional staged update.
The full-operation cell has its own `full_coupled` and `full_update` toggles;
coupling defaults to the staged section's setting.
Updates persist in each in-memory session; recreating a platform reloads the
original runcard.
Rabi amplitude has a runnable example sweep. Other protocols require explicit
sweep inputs, while optional fields retain Qibocal's defaults. For a headless
export, pass these with `--parameters`, for example:

```sh
uv run --group notebooks marimo export html notebooks/calibration.py \
    -o /tmp/ramsey.html -- --protocol ramsey \
    --parameters '{"delay": [40, 4290, 250], "detuning": 1000000}'
```

The notebook loads the example directly with Qibocal's
`create_calibration_platform("split_transmons")`. Its `calibration.json` supplies
the model's initial frequencies, sweetspots, and coherence times through the
standard platform loader. Built-in `Protocol` instances bind parameters directly,
e.g. `protocols.rabi_amplitude(min_amp=0.01,
max_amp=0.28, step_amp=0.016, nshots=4096, relaxation_time=0)`. A separately
created `Executor(platform, targets=[0, 1], update=False)` runs each phase and
returns per-target figures/tables from `executor.report(data, fitted, bound)`.
Updates use `executor.update(fitted, bound)` on the actual session platform.
Connections are caller-managed, and no local protocol adapters are needed.
Display helpers, settings validation, fit-data preparation, and connection
cleanup live in the local `notebooks/calibration_helpers/` package, keeping
the notebook focused on settings and Qibocal calls.

Short coherence windows limit integration cost but do not provide reliable
coherence extraction. Full-window isolated T1/T2 regressions compare fitted times
to the model's scalar parameters. Coupling
shifts transition frequencies: apply spectroscopy before resonant T2, and inspect
goodness-of-fit even when Qibocal reports a successful fit. Qubit spectroscopy's
ideal integrated signal is driven non-ground population, not a resonator response.
No ternary discrimination, resonator calibration, RAW traces, or IQ-cloud
classification is implied by running these protocols.

Finite-shot probabilities can produce zero binomial errors. The staged section
uses a `1/nshots` standard-error floor on a **fit-only copy** of the
data by default; set `error_floor = False` to disable it. Raw acquisition values
are retained. The full-operation section uses Qibocal's fit directly without this
preprocessing. Missing fit results are surfaced rather than presented as
successful calibrations. Both execution paths disconnect in a `finally` block.

The current Qibolab version has a
[headless Marimo shutdown issue](../notebooks/qibolab-platform-shutdown.md):
HTML export writes its output but may not terminate.

The Qibocal module skips when that optional dependency is absent. The tests
workflow has a separate Python 3.12 job for these regressions; the docs workflow
executes the notebook export. The base driver does not depend on Qibocal or marimo.

The implementation has four boundaries: serializable configuration and pulse
compilation; the Qibolab-independent numerical engine; lazy sweep expansion and
bounded scheduling; and terminal measurement conversion. Keep solver jobs
pickleable, random streams tied to experiment coordinates, and unsupported
physical behavior explicit. A new physical approximation needs an accuracy
regression, not just an execution smoke test.

## CI

The [tests workflow](../.github/workflows/tests.yml) exercises Linux on Python 3.11/3.13
and macOS on Python 3.12, including process spawning and distributed execution.
The [docs workflow](../.github/workflows/docs.yml) checks the marimo notebook and
uploads its executed HTML export as the `documentation` artifact.
The [wheels workflow](../.github/workflows/wheels.yml) builds the wheel and sdist
with publishing disabled.

All three workflows run on pushes, pull requests, merge groups, and manual
dispatches. They reuse [qiboteam/workflows](https://github.com/qiboteam/workflows)
on its `uv` ref, like Qibolab and Qibocal. The tests workflow inherits the
`codecov_token` Actions secret for coverage uploads, required for private
repositories.

Documentation remains Markdown in the repository, without a site generator or
deployment pipeline. To generate the notebook documentation locally, run:

```sh
uv sync --locked --group notebooks
uv run --no-dev --with poethepoet poe docs
```

The generated `docs/calibration.html` is ignored by Git.

## Performance

```sh
uv run python benchmarks/rabi.py --points 201 --workers 1 2 --repeats 3
uv run --with qibo python benchmarks/rabi.py --points 21 --workers 1 --legacy
```

The benchmark includes compilation, scheduling, integration, and result conversion,
and reports cold and warm times separately along with analytic error. It is not
a wall-clock CI assertion: process startup, BLAS libraries, cluster communication,
and the size of the physical system change the useful worker count. The optional
legacy comparison runs the emulator in the installed Qibolab release, not a
vendored implementation. Qibolab 0.2.16 needs the additional `qibo` dependency
for its emulator import and expects density matrices in result processing.
The comparison therefore uses the same negligible damping (`T1 = 10^12 ns`)
in both engines and reports it in its metadata; the normal benchmark is closed-system.
