"""Time the performance notebook's sweeps on one backend; write JSON and NPZ.

Run one backend per process, e.g. on a GPU node and on a CPU node of the same
model, then combine the outputs with ``analyze.py``.
"""

import argparse
import json
import os
import platform as system
import sys
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from notebooks.performance import channels, chevron, rabi, ramsey, run  # noqa: E402
from qibolab_simulator import (  # noqa: E402
    DynamiqsSettings,
    Settings,
    SimulatorController,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["qutip", "dynamiqs"], required=True)
    parser.add_argument("--precision", choices=["double", "single"], default="double")
    parser.add_argument(
        "--points", type=int, nargs="+", default=[16, 64, 256, 1024, 4096]
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", required=True, help="Output path, without suffix")
    args = parser.parse_args()

    # Both backends share the tolerances, as in the notebook.
    solver = Settings() if args.precision == "double" else Settings(1e-5, 1e-6)
    options = DynamiqsSettings(precision=args.precision)
    info = {
        "python": system.python_version(),
        "cpus": os.cpu_count() if args.workers > 1 else 1,
        "args": {k: v for k, v in vars(args).items() if k != "out"},
    }
    if args.backend == "dynamiqs":
        import jax

        info["devices"] = [d.device_kind for d in jax.devices()]
    print(json.dumps(info), flush=True)

    scenarios = {"Rabi amplitude": rabi, "Ramsey delay": ramsey, "Chevron": chevron}
    rows, arrays = [], {}
    controller = SimulatorController(
        channels=channels,
        seed=1234,
        solver=solver,
        backend=args.backend,
        dynamiqs=options,
        workers=args.workers,
    )
    controller.connect()
    try:
        for name, scenario in scenarios.items():
            for size in args.points:
                first, values = run(controller, scenario, size)
                warm = [run(controller, scenario, size)[0] for _ in range(args.repeats)]
                row = {
                    "scenario": name,
                    "points": int(values.size),
                    "first": first,
                    "warm": median(warm) if warm else first,
                }
                rows.append(row)
                arrays[f"{name}|{values.size}"] = values
                print(json.dumps(row), flush=True)
                # Rewritten after every sweep, so partial runs keep their results.
                Path(f"{args.out}.json").write_text(
                    json.dumps({"info": info, "rows": rows}, indent=1)
                )
                np.savez(f"{args.out}.npz", **arrays)
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
