import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import os
    import shutil
    from pathlib import Path

    import marimo as mo
    from helpers import show_data, show_reports
    from qibocal import Executor, create_calibration_platform, protocols

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
    The notebook dependencies pin that unmerged PR.

    Edit the settings below. The sections share protocol settings but use separate
    in-memory platforms. Updates last until you rerun platform creation;
    the example runcard is never modified.
    """)
    return


@app.cell
def _():
    path = "var/report/"
    coupled = True

    platform = create_calibration_platform("qutrits")
    if not coupled:
        platform.update({"configs.hamiltonian.couplings": []})

    shutil.rmtree(path)
    executor = Executor.create(platform=platform, path=path, update=False)
    return (executor,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 1. Individual executor calls

    Run acquisition, fitting, and reporting separately, inspecting the outputs
    between stages. Set `coupled = False` in the platform cell for isolated
    calibration.
    """)
    return


@app.cell
def bind_protocol():
    bound = protocols.ramsey(delay=(10, 1000, 100))
    return (bound,)


@app.cell
def acquire_protocol(bound, executor):
    with executor:
        data = executor.acquire(bound, nshots=1000, targets=[0, 1])
    return (data,)


@app.cell
def _(data):
    show_data(data)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Fit

    `error_floor = True` floors binomial errors at `1/nshots` on a fit-only copy.
    Raw data stay unchanged; set it to `False` for unmodified fitting inputs.
    """)
    return


@app.cell
def fit_protocol(bound, data, executor):
    fitted = executor.fit(data, bound, targets=[0, 1])
    fitted
    return (fitted,)


@app.cell
def _(bound, data, executor, fitted):
    reports = executor.report(data, fitted, bound, targets=[0, 1])
    return (reports,)


@app.cell
def _(reports):
    show_reports(reports)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Optional update

    Set `apply_update = True` to use the fit in subsequent acquisitions on this
    platform. Rerun platform creation to restore the original runcard.
    """)
    return


@app.cell
def _(bound, executor, fitted):
    apply_update = False
    if apply_update:
        executor.update(fitted, bound)
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
    return (completed,)


@app.cell
def _(completed):
    show_data(completed.data)
    return


@app.cell
def _(completed):
    completed.results
    return


@app.cell
def display_completed_reports(completed):
    show_reports(completed.reports)
    return


if __name__ == "__main__":
    app.run()
