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
[interactive notebook](../notebooks/calibration.py) use Qibocal's `Executor` and
`BoundProtocol` from
[qiboteam/qibocal#1716](https://github.com/qiboteam/qibocal/pull/1716).
The optional group pins commit `3b15fde3cc229838b093a402cc5e45e3e35e92ed`
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
request. Updates persist across protocols in the current in-memory session;
reset reloads the original runcard. JSON parameter overrides expose each
protocol's own parameters rather than adding another calibration API.

`examples/calibration.py` only prepares the platform, sweep parameters, and bound
protocol; callers execute phases directly with Qibocal's `Executor`. The PR's
built-in routines still take legacy arguments, so small adapters capture the
platform/targets, accept the explicit fit error-floor option, and collect
per-target figures/tables into a report output list. A platform adapter routes
`Executor.update(results)` to the routine's per-target update function: the PR
otherwise calls Qibolab's configuration-dictionary update with fit results.
The notebook runs individual phases so updates remain opt-in; a full
`executor(bound)` workflow also reports (when configured) and applies updates.

Coupled-mode coherence presets use short windows to limit integration cost,
not to claim accurate coherence extraction. Full-window isolated T1/T2
regressions compare fitted times to the model's scalar parameters. Coupling
shifts transition frequencies: apply spectroscopy before resonant T2, and inspect
goodness-of-fit even when Qibocal reports a successful fit. Qubit spectroscopy's
ideal integrated signal is driven non-ground population, not a resonator response.
No ternary discrimination, resonator calibration, RAW traces, or IQ-cloud
classification is implied by running these protocols.

Finite-shot probabilities can produce zero binomial errors. The notebook
explicitly offers a `1/nshots` standard-error floor on a **fit-only copy** of the
data, retaining the raw acquisition values. Failed Qibocal fits are surfaced
rather than treated as successful updates. Platform construction preserves the
notebook runner's signal handlers, and acquisition disconnects in a `finally`
block.

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
