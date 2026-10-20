import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import os
    import platform as system
    from pathlib import Path
    from statistics import median
    from time import perf_counter

    import dynamiqs
    import jax
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import qutip
    from qibolab import (
        AveragingMode,
        Delay,
        ExecutionParameters,
        Parameter,
        Sweeper,
        create_platform,
    )

    from qibolab_simulator import DynamiqsSettings, Settings, SimulatorController

    os.environ["QIBOLAB_PLATFORMS"] = str(
        Path(__file__).resolve().parents[1] / "examples" / "platforms"
    )
    platform = create_platform("qutrits")
    configs = platform.parameters.configs
    natives = platform.natives.single_qubit[0]
    channels = platform.instruments["simulator"].channels

    # Reference categorical slots 1-3, with dashes as a secondary encoding.
    STYLES = {
        "QuTiP": {"color": "#2a78d6", "dash": "solid"},
        "dynamiqs, warm": {"color": "#eb6834", "dash": "solid"},
        "dynamiqs, first call": {"color": "#1baf7a", "dash": "dot"},
    }


@app.cell(hide_code=True)
def _():
    mo.md("""
    # QuTiP and dynamiqs backends

    The simulator solves the same rotating-frame model with either
    [QuTiP](https://qutip.org) or [dynamiqs](https://www.dynamiqs.org), selected by
    `SimulatorController(backend=...)`. Everything else is shared: the runcard,
    pulse compilation, solver tolerances, and the seeded measurement sampling.

    QuTiP integrates each sweep point separately. dynamiqs solves all points with
    the same structure as one batch on the JAX device, so it gains from large
    sweeps and from accelerators. Its first call also compiles the solvers. Later
    calls with the same batch shapes reuse them, even when the swept values
    change.

    This notebook times three sweeps on the two-qutrit example platform (coupled,
    with T1 and T2), checks that the backends agree, and reports the device. Run
    it on a GPU machine to see the accelerator's effect; the defaults take a few
    minutes on a laptop CPU.
    """)
    return


@app.cell(hide_code=True)
def _():
    arguments = mo.cli_args()
    points = [int(p) for p in str(arguments.get("points", "16 64 256")).split()]
    repeats = int(arguments.get("repeats", 1))
    options = DynamiqsSettings(
        method=arguments.get("method", "dopri8"),
        precision=arguments.get("precision", "double"),
    )
    # Single precision cannot meet the default tolerances; both backends share them.
    solver = Settings() if options.precision == "double" else Settings(1e-5, 1e-6)
    mo.md(
        f"""
    ## Settings

    Sweep sizes **{points}**, **{repeats}** warm repeat(s), dynamiqs method
    `{options.method}` in **{options.precision}** precision. Override them with
    `-- --points "16 64" --repeats 3 --method tsit5 --precision single` after
    `marimo edit` or `marimo export`. Chevron sizes are rounded to squares.
    Both backends use `{solver}`.
    """
    )
    return options, points, repeats, solver


@app.cell(hide_code=True)
def _():
    devices = ", ".join(f"`{device}`" for device in jax.devices())
    mo.md(
        f"""
    ## Environment

    | | |
    | --- | --- |
    | JAX devices | {devices} |
    | Python | {system.python_version()} on {system.platform()} |
    | Versions | QuTiP {qutip.__version__}, dynamiqs {dynamiqs.__version__}, JAX {jax.__version__} |
    """
    )
    return


@app.function
def rabi(size):
    """Amplitude sweep of a Gaussian pi pulse: one pulse grid for every point."""
    sequence = natives.RX()
    pulse = sequence[0][1]
    sequence.align(["0/drive", "0/acquisition"])
    sequence += natives.MZ()
    values = np.linspace(0, 0.25, size)
    return sequence, [
        [Sweeper(parameter=Parameter.amplitude, pulses=[pulse], values=values)]
    ]


@app.function
def ramsey(size):
    """Delay sweep between pi/2 pulses: long idle intervals, solved exactly."""
    delay = Delay(duration=0)
    sequence = natives.RX90()
    sequence.append(("0/drive", delay))
    sequence += natives.RX90()
    sequence.align(["0/drive", "0/acquisition"])
    sequence += natives.MZ()
    values = np.linspace(0, 2000, size)
    return sequence, [
        [Sweeper(parameter=Parameter.duration, pulses=[delay], values=values)]
    ]


@app.function
def chevron(size):
    """Frequency and duration sweep: one batch per duration."""
    side = max(2, round(np.sqrt(size)))
    sequence = natives.RX()
    pulse = sequence[0][1]
    sequence.align(["0/drive", "0/acquisition"])
    sequence += natives.MZ()
    frequency = configs["0/drive"].frequency
    return sequence, [
        [
            Sweeper(
                parameter=Parameter.frequency,
                channels=["0/drive"],
                values=frequency + np.linspace(-20e6, 20e6, side),
            )
        ],
        [
            Sweeper(
                parameter=Parameter.duration,
                pulses=[pulse],
                values=np.linspace(20, 200, side).round(),
            )
        ],
    ]


@app.function
def run(controller, scenario, size):
    """Wall time and averaged excited population of one sweep."""
    sequence, sweepers = scenario(size)
    parameters = ExecutionParameters(nshots=1, averaging_mode=AveragingMode.CYCLIC)
    start = perf_counter()
    results = controller.play(configs, [sequence], parameters, sweepers)
    elapsed = perf_counter() - start
    (values,) = results.values()
    return elapsed, values


@app.cell(hide_code=True)
def _(options, points, repeats, solver):
    scenarios = {"Rabi amplitude": rabi, "Ramsey delay": ramsey, "Chevron": chevron}
    rows, curves = [], {}
    with mo.status.progress_bar(
        total=len(scenarios) * len(points), title="Timing sweeps"
    ) as bar:
        for name, scenario in scenarios.items():
            for size in points:
                qutip_controller = SimulatorController(
                    channels=channels, seed=1234, solver=solver
                )
                qutip_time, reference = run(qutip_controller, scenario, size)
                dynamiqs_controller = SimulatorController(
                    channels=channels,
                    seed=1234,
                    solver=solver,
                    backend="dynamiqs",
                    dynamiqs=options,
                )
                first, values = run(dynamiqs_controller, scenario, size)
                warm = median(
                    run(dynamiqs_controller, scenario, size)[0] for _ in range(repeats)
                )
                curves[name, size] = (reference, values)
                rows.append(
                    {
                        "scenario": name,
                        "points": reference.size,
                        "QuTiP (s)": round(qutip_time, 3),
                        "dynamiqs first call (s)": round(first, 3),
                        "dynamiqs warm (s)": round(warm, 3),
                        "warm speedup": round(qutip_time / warm, 2),
                        "max deviation": float(np.max(np.abs(values - reference))),
                    }
                )
                bar.update()
    return curves, rows, scenarios


@app.cell(hide_code=True)
def _(rows, scenarios):
    largest = {row["scenario"]: row for row in rows}
    mo.vstack(
        [
            mo.md("## Results\n\nWarm speedup of dynamiqs at the largest sweep size."),
            mo.hstack(
                [
                    mo.stat(
                        f"{largest[name]['warm speedup']}x",
                        label=name,
                        caption=f"{largest[name]['points']} points",
                    )
                    for name in scenarios
                ],
                justify="start",
            ),
        ]
    )
    return


@app.function
def timing_figure(name, rows):
    """Log-log wall time per backend, labeled directly at the largest size."""
    selected = [row for row in rows if row["scenario"] == name]
    sizes = [row["points"] for row in selected]
    columns = {
        "QuTiP": "QuTiP (s)",
        "dynamiqs, warm": "dynamiqs warm (s)",
        "dynamiqs, first call": "dynamiqs first call (s)",
    }
    figure = go.Figure()
    for series, column in columns.items():
        style = STYLES[series]
        figure.add_scatter(
            x=sizes,
            y=[row[column] for row in selected],
            name=series,
            mode="lines+markers",
            line={"color": style["color"], "dash": style["dash"], "width": 2},
            marker={"size": 8, "color": style["color"]},
            hovertemplate="%{y:.3f} s<extra>" + series + "</extra>",
        )
    times = [row[column] for row in selected for column in columns.values()]
    low, high = np.log10(min(times)), np.log10(max(times))
    # Nudge end labels at least 14 px apart, on a plot area about 200 px tall.
    pixels = 200 / max(high - low + 0.2, 0.5)
    ends = sorted((np.log10(selected[-1][c]), series) for series, c in columns.items())
    previous = -np.inf
    for value, series in ends:
        position = max(value * pixels, previous + 14)
        figure.add_annotation(
            x=np.log10(sizes[-1]),
            y=value,
            yshift=position - value * pixels,
            text=series,
            showarrow=False,
            xanchor="left",
            xshift=8,
            font={"color": "#52514e", "size": 11},
        )
        previous = position
    steps = [m * 10.0**e for e in range(-3, 5) for m in (1, 2, 5)]
    figure.update_layout(
        title={"text": name, "font": {"size": 14}},
        template="simple_white",
        height=340,
        margin={"l": 60, "r": 130, "t": 40, "b": 110},
        legend={"orientation": "h", "yanchor": "top", "y": -0.32},
        hovermode="x unified",
    )
    figure.update_xaxes(
        type="log",
        title="sweep points",
        tickvals=sizes,
        showgrid=True,
        gridcolor="#ecebe8",
    )
    figure.update_yaxes(
        type="log",
        title="wall time (s)",
        tickvals=[t for t in steps if min(times) / 2 <= t <= 2 * max(times)],
        showgrid=True,
        gridcolor="#ecebe8",
    )
    return figure


@app.cell(hide_code=True)
def _(rows, scenarios):
    mo.vstack(
        [
            mo.hstack(
                [mo.ui.plotly(timing_figure(name, rows)) for name in scenarios],
                widths="equal",
                wrap=True,
            ),
            mo.ui.table(rows, selection=None, label="All timings"),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Agreement

    Both backends start from the same model and sample with the same seeds, so
    averaged results differ only by solver error. Below are the Ramsey fringes of
    the largest sweep from each backend, followed by a check of every sweep against
    the default tolerances.
    """)
    return


@app.cell(hide_code=True)
def _(curves, points):
    qutip_fringes, dynamiqs_fringes = curves["Ramsey delay", points[-1]]
    delays = np.linspace(0, 2000, qutip_fringes.size)
    fringes = go.Figure()
    fringes.add_scatter(
        x=delays,
        y=qutip_fringes,
        name="QuTiP",
        mode="lines",
        line={"color": STYLES["QuTiP"]["color"], "width": 2},
    )
    fringes.add_scatter(
        x=delays,
        y=dynamiqs_fringes,
        name="dynamiqs",
        mode="markers",
        marker={
            "color": STYLES["dynamiqs, warm"]["color"],
            "size": 8,
            "line": {"color": "#fcfcfb", "width": 2},
        },
    )
    fringes.update_layout(
        template="simple_white",
        height=320,
        margin={"l": 60, "r": 20, "t": 20, "b": 50},
        legend={"orientation": "h", "y": -0.25},
        hovermode="x unified",
    )
    fringes.update_xaxes(title="delay (ns)", showgrid=True, gridcolor="#ecebe8")
    fringes.update_yaxes(title="excited population", showgrid=True, gridcolor="#ecebe8")
    mo.ui.plotly(fringes)
    return


@app.cell(hide_code=True)
def _(options, rows):
    deviation = max(row["max deviation"] for row in rows)
    # Default tolerances bound both solvers' errors well below this.
    tolerance = 1e-5 if options.precision == "double" else 1e-3
    assert deviation < tolerance, f"Backends disagree by {deviation:.2e}"
    mo.callout(
        mo.md(
            f"Largest deviation between backends: **{deviation:.1e}**, "
            f"below the {tolerance:.0e} tolerance."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Reading the numbers

    - The **first call** includes JIT compilation for each new batch structure and
      size. Batch, sample, and boundary counts are padded to powers of two, so
      nearby sweep sizes share compiled solvers.
    - **Warm** calls are what repeated calibration runs pay. The batch size
      (`batch_size`, default 1024 for dynamiqs) bounds the experiments solved
      together.
    - **Ramsey** idle intervals are exponentiated exactly by dynamiqs, while QuTiP
      integrates them in this coupled, dissipative model.
    - **Chevron** durations change the pulse sampling grid, and each duration
      forms its own batch across frequencies.
    - Single precision is much faster on most GPUs. With `--precision single`,
      both backends use `Settings(rtol=1e-5, atol=1e-6)`, and the agreement
      check is loosened to match.
    """)
    return


if __name__ == "__main__":
    app.run()
