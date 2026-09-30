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
Hatchling for packaging, and one `dev` dependency group; no separate task runner
is needed.

Tests cover analytic dynamics, pulse timing and sweeps, Qibolab `Platform.execute`
and runcard serialization, correlated readout, spawned workers, and a two-process
Dask cluster. The Dask test skips when the optional `distributed` extra is absent.
Run a focused module while iterating, for example `uv run pytest tests/test_sequence.py`.

The implementation has four boundaries: serializable configuration and pulse
compilation; the Qibolab-independent numerical engine; lazy sweep expansion and
bounded scheduling; and terminal measurement conversion. Keep solver jobs
pickleable, random streams tied to experiment coordinates, and unsupported
physical behavior explicit. A new physical approximation needs an accuracy
regression, not just an execution smoke test.

## CI

The [workflow](../.github/workflows/ci.yml) exercises Linux on Python 3.11/3.13
and macOS on Python 3.12, including process spawning and distributed execution.
It runs on every push and pull request without labels or secrets.

Wheel/sdist construction reuses
[`qiboteam/workflows`' uv workflow](https://github.com/qiboteam/workflows/blob/uv/.github/workflows/wheels.yml),
pinned to a reviewed commit, with publishing disabled. That workflow's internal
actions still reference its `uv` branch. The reusable test workflow expects
`poe lint`/`poe test` tasks; the local test job avoids adding a task runner solely
to meet that interface. Documentation is ordinary Markdown in the repository:
there is no documentation generator or deployment pipeline to maintain.

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
