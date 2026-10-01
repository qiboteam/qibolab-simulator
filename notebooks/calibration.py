import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import json
    import os
    from pathlib import Path

    import marimo as mo
    from qibocal import Executor, create_calibration_platform, protocols

    from helpers import (
        acquisition_parameters,
        connected,
        prepare_fit_data,
        require_fit_results,
        show_data,
        show_reports,
    )

    os.environ.setdefault(
        "QIBOLAB_PLATFORMS",
        str(Path(__file__).resolve().parents[1] / "examples" / "platforms"),
    )


@app.cell(hide_code=True)
def _():
    cli = mo.cli_args()
    return (cli,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Qibocal on two split-transmon qutrits

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
    targets = [0, 1]
    nshots = 1000
    return nshots, targets


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
def _():
    coupled = True
    platform = create_calibration_platform("split_transmons")
    if not coupled:
        platform.update({"configs.hamiltonian.couplings": []})
    return coupled, platform


@app.cell
def _(cli):
    routine = protocols.rabi_amplitude
    parameters = {"min_amp": 0.01, "max_amp": 0.28, "step_amp": 0.016}
    if "protocol" in cli:
        routine = getattr(protocols, cli["protocol"])
        if routine is not protocols.rabi_amplitude:
            parameters = {}
    if "parameters" in cli:
        parameters = json.loads(cli["parameters"])
    return parameters, routine


@app.cell(hide_code=True)
def _():
    mo.md("""
    Use `protocols.rabi_amplitude`, `rabi_length`, `ramsey`, `t1`, `t2`,
    `spin_echo`, `qubit_spectroscopy`, or `flux_amplitude_frequency` and supply
    the required sweep fields in `parameters`. Optional fields use Qibocal's
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
def bind_protocol(nshots, parameters, platform, routine, targets):
    _parameters = acquisition_parameters(parameters, nshots, platform, targets)
    bound = routine(**_parameters)
    executor = Executor(platform, targets=targets, update=False)
    return bound, executor


@app.cell
def acquire_protocol(bound, executor, platform):
    with connected(platform):
        data = executor.acquire(bound)
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

    `error_floor = True` floors binomial errors at `1/nshots` on a fit-only copy.
    Raw data stay unchanged; set it to `False` for unmodified fitting inputs.
    """)
    return


@app.cell
def fit_protocol(bound, data, executor, targets):
    error_floor = True
    _fit_data = prepare_fit_data(data, bound.parameters.nshots, error_floor=error_floor)
    fitted = executor.fit(_fit_data, bound)
    require_fit_results(fitted, targets)
    return (fitted,)


@app.cell
def _(fitted):
    fitted
    return


@app.cell
def _(bound, data, executor, fitted):
    reports = executor.report(data, fitted, bound)
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

    `full_executor(bound)` runs all stages and returns a `Completed` object.
    This section uses a fresh platform with the same protocol and targets.

    Override `full_coupled` for a different model, or set `full_update = True`
    for automatic updates. This path uses Qibocal's fitting inputs directly,
    without the error floor.
    """)
    return


@app.cell
def run_protocol(bound, coupled, targets):
    full_coupled = coupled
    full_update = False
    full_platform = create_calibration_platform("split_transmons")
    if not full_coupled:
        full_platform.update({"configs.hamiltonian.couplings": []})
    full_executor = Executor(full_platform, targets=targets, update=full_update)
    with connected(full_platform):
        completed = full_executor(bound)
    require_fit_results(completed.results, targets)
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
def _(completed):
    if completed.reports is not None:
        show_reports(completed.reports)
    return


if __name__ == "__main__":
    app.run()
