import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from examples.calibration import (
        PROTOCOLS,
        acquire,
        apply_update,
        calibration_platform,
        fit,
        report,
    )

    cli = mo.cli_args()
    return (
        PROTOCOLS,
        acquire,
        apply_update,
        calibration_platform,
        cli,
        fit,
        json,
        mo,
        pd,
        report,
    )


@app.cell
def _(mo):
    mo.md("""
    # Qibocal on two split-transmon qutrits

    These are **real Qibocal protocols**, not notebook-generated synthetic traces.
    Qibocal compiles the experiment and executes the simulator through Qibolab.
    Start with Rabi amplitude, then explore duration, coherence, and flux scans.

    The explicit **acquire -> fit -> report -> update** stages are inspired by
    [qiboteam/qibocal#1716](https://github.com/qiboteam/qibocal/pull/1716).
    That PR is unmerged; this notebook uses the published Qibocal **0.2.6**
    routine interface, not its proposed parameter-binding/executor API.

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
def _(acquire, cli, json, mo, platform, settings):
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
    session = acquire(
        _settings["protocol"],
        targets=_settings["targets"],
        points=_settings["points"],
        nshots=_settings["nshots"],
        full_window=_settings["full_window"],
        platform=platform,
        overrides=_overrides,
    )
    mo.md(
        f"**{session.name}** acquired on {session.targets} "
        f"in **{session.seconds:.3f} s**. The controller has been disconnected."
    )
    return (session,)


@app.cell
def _(mo, pd, session):
    mo.accordion(
        {
            "Protocol parameters": mo.md(
                f"```text\n{session.parameters!r}\n```\n\n"
                f"nshots={session.parameters.nshots}; "
                f"relaxation_time={session.parameters.relaxation_time} ns"
            ),
        }
        | {
            f"Raw data: {target}": mo.ui.table(pd.DataFrame(records))
            for target, records in session.data.data.items()
        }
    )
    return


@app.cell
def _(mo, session):
    error_floor = mo.ui.checkbox(
        value=True,
        label="Floor binomial standard errors at 1/nshots on a fit-only copy",
    )
    fit_button = mo.ui.run_button(label=f"Fit {session.name} data with Qibocal")
    mo.vstack([error_floor, fit_button])
    return error_floor, fit_button


@app.cell
def _(cli, error_floor, fit, fit_button, mo, session):
    mo.stop(
        not fit_button.value and "protocol" not in cli,
        mo.md("Inspect the data, then run the fit."),
    )
    fitted = fit(session, error_floor=error_floor.value)
    mo.md(
        f"### Qibocal fit results\n\n```text\n{fitted!r}\n```\n\n"
        f"Fit time: **{session.fit_seconds:.3f} s**. "
        f"Error floor applied to **{session.regularized_points}** points; "
        "the raw acquisition data are unchanged."
    )
    return (fitted,)


@app.cell
def _(fitted, mo, report, session):
    _outputs = []
    for _target, _figures, _table in report(session, fitted):
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
def _(apply_update, fitted, mo, session, update_button):
    mo.stop(not update_button.value, mo.md("Updates are opt-in."))
    apply_update(session, fitted)
    mo.md("""
    Applied Qibocal's update to this **in-memory platform only**.
    Subsequent protocols use these updated parameters; reset the platform to
    reload the original example runcard.
    """)
    return


if __name__ == "__main__":
    app.run()
