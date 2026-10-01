import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    from copy import deepcopy
    from inspect import signature
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

    cli = mo.cli_args()
    return (
        Executor,
        QubitSpectroscopyData,
        create_calibration_platform,
        cli,
        deepcopy,
        json,
        mo,
        np,
        pd,
        protocols,
        signature,
    )


@app.cell
def _(mo):
    mo.md("""
    # Qibocal on two split-transmon qutrits

    These are **real Qibocal protocols**, not notebook-generated synthetic traces.
    Qibocal compiles the experiment and executes the simulator through Qibolab.
    Start with Rabi amplitude, then explore duration, coherence, and flux scans.

    The explicit **acquire -> fit -> report -> update** stages use the
    `Executor` and `BoundProtocol` API from
    [qiboteam/qibocal#1716](https://github.com/qiboteam/qibocal/pull/1716).
    The optional dependencies pin that unmerged PR. Built-in `Protocol`
    instances bind parameters directly, independently of the executor.

    First create a platform session, then acquire protocols against it. Updates
    persist across protocols **within that session**; reset to discard them.
    The example runcard is never modified.
    """)
    return


@app.cell
def _(mo):
    model_settings = mo.ui.checkbox(
        value=True, label="Retain the original 12 MHz exchange coupling"
    ).form(submit_button_label="Create / reset platform")
    model_settings
    return (model_settings,)


@app.cell
def _(cli, create_calibration_platform, mo, model_settings):
    mo.stop(
        model_settings.value is None and "protocol" not in cli,
        mo.md("Create a platform session first."),
    )
    _coupled = model_settings.value if model_settings.value is not None else True
    platform = create_calibration_platform("split_transmons")
    if not _coupled:
        platform.update({"configs.hamiltonian.couplings": []})
    mo.md(
        "**Two-qutrit session ready.** "
        + ("Coupling: 12 MHz." if _coupled else "Explicitly uncoupled.")
    )
    return (platform,)


@app.cell
def _(cli, mo):
    protocol_choice = mo.ui.dropdown(
        options=[
            "rabi_amplitude",
            "rabi_length",
            "ramsey",
            "t1",
            "t2",
            "spin_echo",
            "qubit_spectroscopy",
            "flux_amplitude_frequency",
        ],
        value=cli.get("protocol", "rabi_amplitude"),
        label="Qibocal protocol",
    )
    protocol_choice
    return (protocol_choice,)


@app.cell
def _(mo, platform, protocol_choice, protocols, signature):
    routine = getattr(protocols, protocol_choice.value)
    default_parameters = (
        '{"min_amp": 0.01, "max_amp": 0.28, "step_amp": 0.016}'
        if routine is protocols.rabi_amplitude
        else "{}"
    )
    settings = mo.ui.dictionary(
        {
            "targets": mo.ui.multiselect(
                options={"Qutrit 0": 0, "Qutrit 1": 1},
                value=["Qutrit 0", "Qutrit 1"],
                label="Targets",
            ),
            "nshots": mo.ui.number(
                start=128, stop=16384, step=128, value=4096, label="Shots"
            ),
            "parameters": mo.ui.text_area(
                value=default_parameters,
                label="Qibocal acquisition parameters (JSON)",
            ),
        }
    ).form(submit_button_label=f"Acquire on {platform.name}")
    mo.vstack(
        [
            mo.md(
                "Supply the required sweep fields; optional fields use Qibocal's "
                f"defaults.\n\n```text\n{routine.parameters_type.__name__}"
                f"{signature(routine.parameters_type)}\n```"
            ),
            settings,
            mo.md("""
            Acquisition runs **only on submission**. Coupled long-delay experiments
            can take substantially longer; reset without coupling for isolated
            single-qubit calibration. Both modes still simulate two qutrits.

            Short coherence windows are demonstrations, not reliable T1/T2
            estimates. For decay fits, specify roughly two decay times,
            preferably uncoupled; long coupled delays can be expensive.
            Increase resolution after checking the raw data.
            Coupling shifts resonances: fit/apply spectroscopy before resonant T2.
            Successful fits can still have poor goodness-of-fit; inspect the plots
            and uncertainties rather than treating them as accurate calibrations.
            Readout is ideal binary non-ground discrimination: resonator/RAW
            protocols and genuine three-state assignment are not supported.
            """),
        ]
    )
    return default_parameters, routine, settings


@app.cell
def _(
    Executor,
    cli,
    default_parameters,
    json,
    mo,
    platform,
    routine,
    settings,
):
    _settings = settings.value
    if _settings is None and "protocol" in cli:
        _settings = {
            "targets": [0, 1],
            "nshots": 4096,
            "parameters": cli.get("parameters", default_parameters),
        }
    mo.stop(_settings is None, mo.md("Submit a protocol to acquire its data."))
    targets = _settings["targets"]
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Select one or both distinct qutrit targets")
    _parameters = json.loads(_settings["parameters"])
    if not isinstance(_parameters, dict):
        raise ValueError("Acquisition parameters must be a JSON object")
    _parameters = {"nshots": _settings["nshots"], "relaxation_time": 0} | _parameters
    if (
        isinstance(_parameters["nshots"], bool)
        or not isinstance(_parameters["nshots"], int)
        or _parameters["nshots"] < 1
    ):
        raise ValueError("Protocol nshots must be a positive integer")
    bound = routine(**_parameters)
    executor = Executor(platform, targets=targets, update=False)
    try:
        platform.connect()
        data = executor.acquire(bound)
    finally:
        platform.disconnect()
    mo.md(f"Acquired on **{targets}**. The controller has been disconnected.")
    return bound, data, executor, targets


@app.cell
def _(QubitSpectroscopyData, bound, data, mo, pd):
    _tables = {}
    for _target, _records in data.data.items():
        if isinstance(data, QubitSpectroscopyData):
            _frame = pd.DataFrame(
                {
                    "frequency": data.drive_frequencies[_target],
                    "signal": data.signal(_target),
                    "phase": data.phase(_target),
                }
            )
        else:
            _frame = pd.DataFrame(_records)
        _tables[f"Raw data: {_target}"] = mo.ui.table(_frame)
    mo.accordion(
        {
            "Protocol parameters": mo.md(
                f"```text\n{bound.parameters!r}\n```\n\n"
                f"nshots={bound.parameters.nshots}; "
                f"relaxation_time={bound.parameters.relaxation_time} ns"
            ),
        }
        | _tables
    )
    return


@app.cell
def _(data, mo):
    error_floor = mo.ui.checkbox(
        value=True,
        label="Floor binomial standard errors at 1/nshots on a fit-only copy",
    )
    fit_button = mo.ui.run_button(label=f"Fit {type(data).__name__} with Qibocal")
    mo.vstack([error_floor, fit_button])
    return error_floor, fit_button


@app.cell
def _(
    bound,
    cli,
    data,
    deepcopy,
    error_floor,
    executor,
    fit_button,
    mo,
    np,
    targets,
):
    mo.stop(
        not fit_button.value and "protocol" not in cli,
        mo.md("Inspect the data, then run the fit."),
    )
    _fit_data = deepcopy(data)
    if error_floor.value:
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
    mo.md(
        f"### Qibocal fit results\n\n```text\n{fitted!r}\n```\n\n"
        f"Error floor: **{'enabled' if error_floor.value else 'disabled'}**; "
        "the raw acquisition data are unchanged."
    )
    return (fitted,)


@app.cell
def _(bound, data, executor, fitted, mo):
    _reports = executor.report(data, fitted, bound)
    _outputs = []
    for _target, (_figures, _table) in _reports.items():
        _outputs.append(mo.md(f"### Target {_target}"))
        _outputs.extend(mo.ui.plotly(_figure) for _figure in _figures)
        if _table:
            _outputs.append(mo.Html(_table))
    mo.vstack(_outputs)
    return


@app.cell
def _(fitted, mo):
    update_button = mo.ui.run_button(
        label=f"Apply {type(fitted).__name__} to the session platform"
    )
    update_button
    return (update_button,)


@app.cell
def _(bound, executor, fitted, mo, update_button):
    mo.stop(not update_button.value, mo.md("Updates are opt-in."))
    executor.update(fitted, bound)
    mo.md("""
    Applied Qibocal's update to this **in-memory platform only**.
    Subsequent protocols use these updated parameters; reset the platform to
    reload the original example runcard.
    """)
    return


if __name__ == "__main__":
    app.run()
