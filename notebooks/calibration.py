import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from dataclasses import replace
    from pathlib import Path
    from time import perf_counter

    import marimo as mo
    import pandas as pd
    from qibocal.protocols.qubit_spectroscopies.qubit_spectroscopy import (
        QubitSpectroscopyData,
    )

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from examples.calibration import (
        PROTOCOLS,
        bind,
        calibration_platform,
    )

    cli = mo.cli_args()
    return (
        PROTOCOLS,
        QubitSpectroscopyData,
        bind,
        calibration_platform,
        cli,
        json,
        mo,
        pd,
        perf_counter,
        replace,
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
    The optional dependencies pin that unmerged PR. Small adapters bind the
    built-in routines' legacy platform/target arguments and collect reports.

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
def _(calibration_platform, cli, mo, model_settings):
    mo.stop(
        model_settings.value is None and "protocol" not in cli,
        mo.md("Create a platform session first."),
    )
    _coupled = model_settings.value if model_settings.value is not None else True
    platform = calibration_platform(coupled=_coupled)
    mo.md(
        "**Two-qutrit session ready.** "
        + ("Coupling: 12 MHz." if _coupled else "Explicitly uncoupled.")
    )
    return (platform,)


@app.cell
def _(PROTOCOLS, mo, platform):
    settings = mo.ui.dictionary(
        {
            "protocol": mo.ui.dropdown(
                options=list(PROTOCOLS),
                value="rabi_amplitude",
                label="Qibocal protocol",
            ),
            "targets": mo.ui.multiselect(
                options={"Qutrit 0": 0, "Qutrit 1": 1},
                value=["Qutrit 0", "Qutrit 1"],
                label="Targets",
            ),
            "points": mo.ui.slider(
                start=9, stop=41, step=4, value=17, label="Sweep resolution"
            ),
            "nshots": mo.ui.number(
                start=128, stop=16384, step=128, value=4096, label="Shots"
            ),
            "full_window": mo.ui.checkbox(
                value=False, label="Use full coherence windows (potentially slow)"
            ),
            "overrides": mo.ui.text_area(
                value="{}",
                label='Qibocal parameter overrides (JSON, e.g. {"drive_duration": 200})',
            ),
        }
    ).form(submit_button_label=f"Acquire on {platform.name}")
    mo.vstack(
        [
            settings,
            mo.md("""
            Acquisition runs **only on submission**. Coupled long-delay experiments
            can take substantially longer; reset without coupling for isolated
            single-qubit calibration. Both modes still simulate two qutrits.

            The default coherence sweeps cover only about 4 microseconds:
            **short-window demonstrations**, not reliable T1/T2 estimates.
            Enable full windows for roughly two decay times, preferably uncoupled.
            Increase resolution after checking the raw data.
            Coupling shifts resonances: fit/apply spectroscopy before resonant T2.
            Successful fits can still have poor goodness-of-fit; inspect the plots
            and uncertainties rather than treating them as accurate calibrations.
            Readout is ideal binary non-ground discrimination: resonator/RAW
            protocols and genuine three-state assignment are not supported.
            """),
        ]
    )
    return (settings,)


@app.cell
def _(bind, cli, json, mo, perf_counter, platform, settings):
    _settings = settings.value
    if _settings is None and "protocol" in cli:
        _settings = {
            "protocol": cli["protocol"],
            "targets": [0, 1],
            "points": 17,
            "nshots": 4096,
            "full_window": False,
            "overrides": "{}",
        }
    mo.stop(_settings is None, mo.md("Submit a protocol to acquire its data."))
    _overrides = json.loads(_settings["overrides"])
    if not isinstance(_overrides, dict):
        raise ValueError("Parameter overrides must be a JSON object")
    executor, bound = bind(
        _settings["protocol"],
        targets=_settings["targets"],
        points=_settings["points"],
        nshots=_settings["nshots"],
        full_window=_settings["full_window"],
        platform=platform,
        overrides=_overrides,
        reports=[],
    )
    _start = perf_counter()
    data = executor.acquire(bound)
    acquisition_seconds = perf_counter() - _start
    mo.md(
        f"**{_settings['protocol']}** acquired on {_settings['targets']} "
        f"in **{acquisition_seconds:.3f} s**. The controller has been disconnected."
    )
    return bound, data, executor


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
def _(bound, cli, data, error_floor, executor, fit_button, mo, perf_counter, replace):
    mo.stop(
        not fit_button.value and "protocol" not in cli,
        mo.md("Inspect the data, then run the fit."),
    )
    fit_bound = replace(bound, fitpars=error_floor.value)
    _start = perf_counter()
    fitted = executor.fit(data, fit_bound)
    _seconds = perf_counter() - _start
    mo.md(
        f"### Qibocal fit results\n\n```text\n{fitted!r}\n```\n\n"
        f"Fit time: **{_seconds:.3f} s**. "
        f"Error floor: **{'enabled' if error_floor.value else 'disabled'}**; "
        "the raw acquisition data are unchanged."
    )
    return (fitted,)


@app.cell
def _(bound, data, executor, fitted, mo, replace):
    _reports = []
    executor.report(data, fitted, replace(bound, reportpars=_reports))
    _outputs = []
    for _target, _figures, _table in _reports:
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
def _(executor, fitted, mo, update_button):
    mo.stop(not update_button.value, mo.md("Updates are opt-in."))
    executor.update(fitted)
    mo.md("""
    Applied Qibocal's update to this **in-memory platform only**.
    Subsequent protocols use these updated parameters; reset the platform to
    reload the original example runcard.
    """)
    return


if __name__ == "__main__":
    app.run()
