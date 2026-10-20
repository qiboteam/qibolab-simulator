"""Combine the benchmark runs in ``results.zip`` into a table and log-log charts."""

import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).parent
RESULTS = HERE / "results.zip"
SCENARIOS = ["Rabi amplitude", "Ramsey delay", "Chevron"]
# Label, file stem, color (reference categorical slots), and dash.
RUNS = [
    ("QuTiP, 1 process", "qutip-double", "#2a78d6", "solid"),
    ("QuTiP, 32 processes", "qutip-double-32", "#2a78d6", "dot"),
    ("dynamiqs GPU, double", "dynamiqs-double", "#eb6834", "solid"),
    ("QuTiP, 1 process, loose tol.", "qutip-single", "#1baf7a", "dash"),
    ("dynamiqs GPU, single", "dynamiqs-single", "#eda100", "solid"),
]


def load():
    rows, arrays = {}, {}
    with ZipFile(RESULTS) as archive:
        for label, stem, *_ in RUNS:
            content = json.loads(archive.read(f"{stem}.json"))
            rows[label] = {(r["scenario"], r["points"]): r for r in content["rows"]}
            arrays[label] = dict(np.load(io.BytesIO(archive.read(f"{stem}.npz"))))
    return rows, arrays


def seconds(value):
    return "–" if value is None else f"{value:.2f}"


def ratio(numerator, denominator):
    if numerator is None or denominator is None:
        return "–"
    return f"**{numerator / denominator:.1f}×**"


def deviation(arrays, first, second, key):
    if key not in arrays[first] or key not in arrays[second]:
        return "–"
    return f"{np.max(np.abs(arrays[first][key] - arrays[second][key])):.1e}"


def table(rows, arrays):
    def warm(label, scenario, size, field="warm"):
        row = rows[label].get((scenario, size))
        return None if row is None else row[field]

    lines = [
        "| Scenario | Points | QuTiP 1 proc. | QuTiP 32 proc. "
        "| GPU double, first / warm | Speedup | QuTiP loose tol. "
        "| GPU single, first / warm | Speedup | Max deviation, double / single |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in SCENARIOS:
        sizes = sorted({k[1] for r in rows.values() for k in r if k[0] == scenario})
        for size in sizes:
            key = f"{scenario}|{size}"
            serial = warm("QuTiP, 1 process", scenario, size)
            parallel = warm("QuTiP, 32 processes", scenario, size)
            gpu = warm("dynamiqs GPU, double", scenario, size)
            gpu_first = warm("dynamiqs GPU, double", scenario, size, "first")
            loose = warm("QuTiP, 1 process, loose tol.", scenario, size)
            single = warm("dynamiqs GPU, single", scenario, size)
            single_first = warm("dynamiqs GPU, single", scenario, size, "first")
            double_dev = deviation(
                arrays, "QuTiP, 1 process", "dynamiqs GPU, double", key
            )
            single_dev = deviation(
                arrays, "QuTiP, 1 process, loose tol.", "dynamiqs GPU, single", key
            )
            lines.append(
                f"| {scenario} | {size} | {seconds(serial)} | {seconds(parallel)} "
                f"| {seconds(gpu_first)} / {seconds(gpu)} | {ratio(serial, gpu)} "
                f"| {seconds(loose)} | {seconds(single_first)} / {seconds(single)} "
                f"| {ratio(loose, single)} | {double_dev} / {single_dev} |"
            )
    return "\n".join(lines) + "\n"


def chart(rows, scenario, path):
    figure = go.Figure()
    ends, times = [], []
    for label, _, color, dash in RUNS:
        points = sorted(
            (k[1], v["warm"]) for k, v in rows[label].items() if k[0] == scenario
        )
        if not points:
            continue
        x, y = zip(*points, strict=True)
        times += y
        figure.add_scatter(
            x=x,
            y=y,
            name=label,
            mode="lines+markers",
            line={"color": color, "dash": dash, "width": 2},
            marker={"size": 8, "color": color},
        )
        ends.append((np.log10(y[-1]), label, x[-1]))
    low, high = np.log10(min(times)), np.log10(max(times))
    # Nudge direct labels at least 15 px apart, on a plot area about 300 px tall.
    pixels = 300 / max(high - low + 0.2, 0.5)
    previous = -np.inf
    for value, label, end in sorted(ends):
        position = max(value * pixels, previous + 15)
        figure.add_annotation(
            x=np.log10(end),
            y=value,
            yshift=position - value * pixels,
            text=label,
            showarrow=False,
            xanchor="left",
            xshift=8,
            font={"color": "#52514e", "size": 11},
        )
        previous = position
    steps = [m * 10.0**e for e in range(-3, 5) for m in (1, 2, 5)]
    sizes = sorted({k[1] for r in rows.values() for k in r if k[0] == scenario})
    figure.update_layout(
        title={
            "text": f"{scenario}: warm wall time, RTX A6000 vs EPYC 7773X",
            "font": {"size": 14},
        },
        template="simple_white",
        width=760,
        height=440,
        margin={"l": 70, "r": 210, "t": 50, "b": 120},
        legend={"orientation": "h", "yanchor": "top", "y": -0.22},
    )
    figure.update_xaxes(
        type="log", title="sweep points", tickvals=sizes, gridcolor="#ecebe8"
    )
    figure.update_yaxes(
        type="log",
        title="wall time (s)",
        gridcolor="#ecebe8",
        tickvals=[t for t in steps if min(times) / 2 <= t <= 2 * max(times)],
    )
    figure.write_image(path, scale=2)


def main():
    rows, arrays = load()
    (HERE / "results.md").write_text(table(rows, arrays))
    for scenario in SCENARIOS:
        name = scenario.split()[0].lower()
        chart(rows, scenario, HERE / f"{name}.png")


if __name__ == "__main__":
    main()
