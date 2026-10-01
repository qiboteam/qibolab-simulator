import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import json
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
    # Qibocal on two split-transmon qutrits

    Qibocal runs real calibration protocols through the Qibolab simulator.
    Explore Ramsey in individual stages and qubit flux in a full operation.

    Compare **acquire -> fit -> report -> update** with a single `executor(bound)`
    call, using the `Executor` and `BoundProtocol` API from
    [qiboteam/qibocal#1716](https://github.com/qiboteam/qibocal/pull/1716).
    The notebook dependencies track that unmerged PR's branch.

    Edit the settings below. Both sections share an executor and in-memory
    platform. Updates last until you rerun platform creation; the example
    runcard is never modified. Calibration output is written to `var/report/`.
    """)
    return


@app.cell
def create_executor():
    path = "var/report/"
    coupled = True

    platform = create_calibration_platform("split_transmons")
    if not coupled:
        platform.update({"configs.hamiltonian.couplings": []})

    if Path(path).exists():
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


@app.cell(hide_code=True)
def _():
    mo.md("""
    Use `protocols.rabi_amplitude`, `rabi_length`, `ramsey`, `t1`, `t2`,
    `spin_echo`, `qubit_spectroscopy`, or `flux_amplitude_frequency` and supply
    the required sweep fields in the protocol call. Optional fields use Qibocal's
    defaults. CLI `--protocol` and `--parameters` overrides support headless export.

    Both modes simulate two qutrits. For reliable T1/T2 fits, scan roughly two
    decay times; long coupled delays can be expensive. Inspect raw data before
    increasing resolution, and check fit uncertainties. Coupling shifts
    resonances: apply spectroscopy before resonant T2.

    Readout is ideal binary non-ground discrimination: resonator/RAW protocols
    and genuine three-state assignment are not supported.
    """)
    return


@app.cell
def bind_protocol():
    bound = protocols.ramsey(delay=(10, 1000, 100))
    _cli = mo.cli_args()
    if "protocol" in _cli:
        _routine = getattr(protocols, _cli["protocol"])
        _parameters = (
            {"min_amp": 0.01, "max_amp": 0.28, "step_amp": 0.016}
            if _routine is protocols.rabi_amplitude
            else {}
        )
        _parameters.update(json.loads(_cli.get("parameters", "{}")))
        bound = _routine(**_parameters)
    return (bound,)


@app.cell
def acquire_protocol(bound, executor):
    with executor:
        data = executor.acquire(bound, nshots=1000, targets=[0, 1])
    return (data,)


@app.cell
def _(bound):
    bound.parameters
    return


@app.cell
def _(data):
    show_data(data)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Fit

    Pass the acquisition data directly to Qibocal's fit, then inspect the fitted
    results and uncertainties before applying an update.
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
        executor.update(fitted, bound, targets=[0, 1])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 2. Full operation with `Executor.__call__()`

    `executor(bound)` runs all stages, including report generation, and returns
    a `Completed` object.
    This section reuses the executor and platform from section 1 for a qubit
    flux scan on target 0. Set `update = True` in executor creation for automatic
    updates. A CLI `--protocol` override runs the bound protocol in both sections.
    """)
    return


@app.cell
def run_protocol(bound, executor):
    _protocol = (
        bound
        if "protocol" in mo.cli_args()
        else protocols.qubit_flux(
            frequency=("linspace", 4.78e9, 4.83e9, 40),
            bias=("linspace", 0.35, 0.46, 30),
            drive_duration=1000,
        )
    )
    with executor:
        completed = executor(
            _protocol,
            nshots=4096,
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
