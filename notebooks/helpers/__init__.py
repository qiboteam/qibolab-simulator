from contextlib import contextmanager
from copy import deepcopy

import marimo as mo
import numpy as np
import pandas as pd
from qibocal.protocols.qubit_spectroscopies.qubit_spectroscopy import (
    QubitSpectroscopyData,
)


def acquisition_parameters(parameters, nshots, platform, targets):
    """Validate notebook settings and fill in acquisition defaults."""
    if (
        not targets
        or len(set(targets)) != len(targets)
        or any(target not in platform.qubits for target in targets)
    ):
        raise ValueError("Select one or both distinct qutrit targets")
    if not isinstance(parameters, dict):
        raise ValueError("Acquisition parameters must be a dictionary")
    parameters = {"nshots": nshots, "relaxation_time": 0} | parameters
    shots = parameters["nshots"]
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("Protocol nshots must be a positive integer")
    return parameters


@contextmanager
def connected(platform):
    """Disconnect even when acquisition or fitting fails."""
    try:
        platform.connect()
        yield platform
    finally:
        platform.disconnect()


def prepare_fit_data(data, nshots, *, error_floor=True):
    """Apply an optional binomial error floor without changing raw data."""
    fit_data = deepcopy(data)
    if error_floor:
        for records in fit_data.data.values():
            if records.dtype.names is not None and "error" in records.dtype.names:
                records["error"] = np.maximum(records["error"], 1 / nshots)
    return fit_data


def require_fit_results(results, targets):
    """Stop before reporting or updating targets whose fit failed."""
    missing = [target for target in targets if results is None or target not in results]
    if missing:
        raise ValueError(
            f"Qibocal fit has no results for {missing}; "
            "inspect the acquisition data and Qibocal logs"
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
