import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import os
    import shutil
    import signal
    from pathlib import Path

    import marimo as mo
    from qibocal import create_calibration_platform, protocols
    from qibocal_widgets import Executor
    from qibocal_widgets.marimo import Marimo

    os.environ["QIBOLAB_PLATFORMS"] = str(
        Path(__file__).resolve().parents[1] / "examples" / "platforms"
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Qibocal on two qutrits

    Qibocal runs real calibration protocols through the Qibolab simulator.
    Start with Rabi amplitude, then explore duration, coherence, or flux scans.

    Compare **acquire -> fit -> report -> update** with a single `executor(bound)`
    call, using the `Executor` and `BoundProtocol` API from
    [qiboteam/qibocal#1716](https://github.com/qiboteam/qibocal/pull/1716).
    The notebook dependencies pin Qibocal's `protodev` branch, which includes it.

    Edit the settings below. The sections share an in-memory platform and executor.
    Updates last until you rerun platform creation; the example runcard is never
    modified.
    """)
    return


@app.cell
def _():
    path = "var/report/"
    coupled = True

    # Qibolab platforms replace the termination handlers, which marimo needs to
    # stop its kernel: headless exports would never exit.
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    platform = create_calibration_platform("qutrits")
    for number, handler in handlers.items():
        signal.signal(number, handler)
    if not coupled:
        platform.update({"configs.hamiltonian.couplings": []})

    if Path(path).exists():
        shutil.rmtree(path)
    executor = Executor.create(
        platform=platform, path=path, update=False, frontends=[Marimo()]
    )
    return (executor,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 1. Individual executor calls

    Run acquisition, fitting, and reporting separately, inspecting the outputs
    between stages. Each stage returns a `Completed` object containing the bound
    protocol, targets, and outputs collected so far. Set `coupled = False` in the
    platform cell for isolated calibration.
    """)
    return


@app.cell
def bind_protocol():
    bound = protocols.ramsey(delay=(10, 1000, 100))
    return (bound,)


@app.cell
def acquire_protocol(bound, executor):
    with executor:
        acquired = executor.acquire(bound, nshots=1000, targets=[0, 1])
    acquired
    return (acquired,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Fit

    Pass the acquired `Completed` object to `executor.fit()`. It returns a new
    `Completed` object with fitting results attached, retaining the data, bound
    protocol, and targets.
    """)
    return


@app.cell
def fit_protocol(acquired, executor):
    fitted = executor.fit(acquired)
    fitted.results
    return (fitted,)


@app.cell
def _(executor, fitted):
    reported = executor.report(fitted)
    reported
    return (reported,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Optional update

    Set `apply_update = True` to use the fit in subsequent acquisitions on this
    platform. Rerun platform creation to restore the original runcard.
    """)
    return


@app.cell
def _(executor, reported):
    apply_update = False
    if apply_update:
        executor.update(reported)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 2. Full operation with `Executor.__call__()`

    `executor(bound)` runs all stages, including report generation, and returns
    a `Completed` object.
    This section uses a fresh platform with the same protocol and targets.

    Override `coupled` in the function below for a different model, or set
    `update = True` for automatic updates. Function scope keeps this platform
    and executor separate from section 1. This path uses Qibocal's fitting
    inputs directly, without the error floor.
    """)
    return


@app.cell
def run_protocol(executor):
    with executor:
        completed = executor(
            protocols.qubit_flux(
                frequency=("linspace", 4.78e9, 4.83e9, 40),
                bias=("linspace", 0.35, 0.46, 30),
                drive_duration=1000,
            ),
            targets=[0],
        )
    completed
    return (completed,)


@app.cell
def _(completed):
    completed.results
    return


if __name__ == "__main__":
    app.run()
