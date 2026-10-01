import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import json
    import os
    from copy import deepcopy
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    from qibocal import Executor, create_calibration_platform, protocols
    from qibocal.protocols.qubit_spectroscopies.qubit_spectroscopy import (
        QubitSpectroscopyData,
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

    These are **real Qibocal protocols**, not notebook-generated synthetic traces.
    Qibocal compiles the experiment and executes the simulator through Qibolab.
    Start with Rabi amplitude, then explore duration, coherence, and flux scans.

    Both the explicit **acquire -> fit -> report -> update** stages and the
    full-operation `executor(bound)` call use the
    `Executor` and `BoundProtocol` API from
    [qiboteam/qibocal#1716](https://github.com/qiboteam/qibocal/pull/1716).
    The optional dependencies pin that unmerged PR. Built-in `Protocol`
    instances bind parameters directly, independently of the executor.

    Edit the Python settings and protocol call below, then run the cells.
    The two sections use separate in-memory platforms and share protocol settings.
    Updates persist **within the in-memory platform session**; rerun the platform
    creation cell to discard them. The example runcard is never modified.
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

    Both coupled and isolated modes simulate two qutrits.
    For decay fits, specify roughly two decay times:
    short windows are demonstrations, not reliable T1/T2 estimates, and long
    coupled delays can be expensive. Increase resolution after inspecting raw
    data. Coupling shifts resonances: fit/apply spectroscopy before resonant T2.
    Inspect plots and uncertainties even when a fit succeeds.

    Readout is ideal binary non-ground discrimination: resonator/RAW protocols
    and genuine three-state assignment are not supported.
    """)
    return


@app.cell
def bind_protocol(nshots, parameters, platform, routine, targets):
    if (
        not targets
        or len(set(targets)) != len(targets)
        or any(target not in platform.qubits for target in targets)
    ):
        raise ValueError("Select one or both distinct qutrit targets")
    if not isinstance(parameters, dict):
        raise ValueError("Acquisition parameters must be a dictionary")
    _parameters = {"nshots": nshots, "relaxation_time": 0} | parameters
    if (
        isinstance(_parameters["nshots"], bool)
        or not isinstance(_parameters["nshots"], int)
        or _parameters["nshots"] < 1
    ):
        raise ValueError("Protocol nshots must be a positive integer")
    bound = routine(**_parameters)
    executor = Executor(platform, targets=targets, update=False)
    return bound, executor


@app.cell
def acquire_protocol(bound, executor, platform):
    try:
        platform.connect()
        data = executor.acquire(bound)
    finally:
        platform.disconnect()
    return (data,)


@app.cell
def _():
    def show_data(acquired):
        tables = {}
        for target, records in acquired.data.items():
            if isinstance(acquired, QubitSpectroscopyData):
                frame = pd.DataFrame(
                    {
                        "frequency": acquired.drive_frequencies[target],
                        "signal": acquired.signal(target),
                        "phase": acquired.phase(target),
                    }
                )
            else:
                frame = pd.DataFrame(records)
            tables[f"Raw data: {target}"] = mo.ui.table(frame)
        return mo.accordion(tables)

    def show_reports(reports):
        outputs = {}
        for target, (figures, table) in reports.items():
            plots = [mo.ui.plotly(figure) for figure in figures]
            if table:
                plots.append(mo.Html(table))
            outputs[f"Target {target}"] = mo.vstack(plots)
        return mo.accordion(outputs)

    return show_data, show_reports


@app.cell
def _(bound):
    bound.parameters
    return


@app.cell
def _(data, show_data):
    show_data(data)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Fit

    `error_floor = True` floors binomial standard errors at `1/nshots` on a
    fit-only copy. Raw acquisition data are unchanged. Set it to `False` in the
    fit cell to use Qibocal's unmodified fitting inputs.
    """)
    return


@app.cell
def fit_protocol(bound, data, executor, targets):
    error_floor = True
    _fit_data = deepcopy(data)
    if error_floor:
        for _records in _fit_data.data.values():
            if _records.dtype.names is not None and "error" in _records.dtype.names:
                _records["error"] = np.maximum(
                    _records["error"], 1 / bound.parameters.nshots
                )
    fitted = executor.fit(_fit_data, bound)
    _missing = [target for target in targets if target not in fitted]
    if _missing:
        raise ValueError(
            f"Qibocal fit has no results for {_missing}; "
            "inspect the acquisition data and Qibocal logs"
        )
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
def _(reports, show_reports):
    show_reports(reports)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Optional update

    Set `apply_update = True` in the next cell to apply the fit to this
    in-memory platform only. Subsequent acquisitions use the updated parameters.
    Rerun the platform creation cell to reload the original runcard.
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

    `full_executor(bound)` acquires, fits, reports, and optionally updates in one
    call, returning a `Completed` object. This example uses a fresh platform and
    the same protocol parameters and targets as above.

    `full_coupled` defaults to the staged section's `coupled` setting; override it
    for a different model. Set `full_update = True` to enable automatic updates.
    Both settings are in the execution cell. This path uses Qibocal's fitting
    inputs directly, without the staged example's fit-only error floor.
    Inspect results and uncertainties.
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
    try:
        full_platform.connect()
        completed = full_executor(bound)
    finally:
        full_platform.disconnect()
    if completed.results is None or any(
        target not in completed.results for target in targets
    ):
        raise ValueError(
            "Qibocal full operation has missing fit results; "
            "inspect the acquisition data and Qibocal logs"
        )
    return (completed,)


@app.cell
def _(completed, show_data):
    show_data(completed.data)
    return


@app.cell
def _(completed):
    completed.results
    return


@app.cell
def _(completed, show_reports):
    if completed.reports is not None:
        show_reports(completed.reports)
    return


if __name__ == "__main__":
    app.run()
