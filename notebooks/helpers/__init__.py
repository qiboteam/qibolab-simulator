import marimo as mo
import pandas as pd
from qibocal.protocols.qubit_spectroscopies.qubit_spectroscopy import (
    QubitSpectroscopyData,
)


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
