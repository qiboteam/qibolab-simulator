"""Reproducible end-to-end Rabi benchmark; emits timings and analytic error."""

import argparse
import json
import platform as system
from time import perf_counter

import numpy as np
import qibolab
import qutip
from qibolab import (
    Acquisition,
    AveragingMode,
    ExecutionParameters,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
)

from qibolab_simulator import (
    DriveConfig,
    HamiltonianConfig,
    QubitConfig,
    ReadoutConfig,
    SimulatorController,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=201)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Also time Qibolab's installed lab-frame emulator",
    )
    args = parser.parse_args()
    if min(args.points, args.repeats, args.batch_size, *args.workers) < 1:
        parser.error("counts must be positive")
    # Released Qibolab expects density matrices. Use the same negligible damping
    # in both engines for this comparison, rather than patching its ket handling.
    t1 = 1e12 if args.legacy else None
    configs = {
        "hamiltonian": HamiltonianConfig(qubits={0: QubitConfig(frequency=5e9, t1=t1)}),
        "0/drive": DriveConfig(target=0, frequency=5e9, rabi_frequency=12.5e6),
        "0/acquisition": ReadoutConfig(target=0),
    }
    pulse = Pulse(duration=40, amplitude=1, envelope=Rectangular())
    acq = Acquisition(duration=100)
    sequence = PulseSequence([("0/drive", pulse)])
    sequence.align(["0/drive", "0/acquisition"])
    sequence.append(("0/acquisition", acq))
    amplitudes = np.linspace(0, 1, args.points)
    sweeper = Sweeper(parameter=Parameter.amplitude, pulses=[pulse], values=amplitudes)
    options = ExecutionParameters(averaging_mode=AveragingMode.CYCLIC)
    reference = np.sin(np.pi * amplitudes / 2) ** 2
    print(
        json.dumps(
            {
                "python": system.python_version(),
                "platform": system.platform(),
                "qibolab": qibolab.__version__,
                "qutip": qutip.__version__,
                "points": args.points,
                "batch_size": args.batch_size,
                "t1_ns": t1,
            }
        )
    )
    if args.legacy:
        from qibolab import AcquisitionConfig
        from qibolab.instruments.emulator import DriveEmulatorConfig, EmulatorController
        from qibolab.instruments.emulator import HamiltonianConfig as LegacyHamiltonian

        legacy_configs = {
            "hamiltonian": LegacyHamiltonian(
                qubits={0: {"frequency": 5e9, "t1": {(0, 1): t1}}}
            ),
            "0/drive": DriveEmulatorConfig(frequency=5e9, rabi_frequency=12.5e6),
            "0/acquisition": AcquisitionConfig(delay=0, smearing=0),
        }
        legacy = EmulatorController(address="local")
        times = []
        for _ in range(args.repeats + 1):
            start = perf_counter()
            values = legacy.play(legacy_configs, [sequence], options, [[sweeper]])[
                acq.id
            ]
            times.append(perf_counter() - start)
        print(
            json.dumps(
                {
                    "engine": "qibolab-emulator-lab-frame",
                    "cold_seconds": times[0],
                    "warm_median_seconds": float(np.median(times[1:])),
                    "max_absolute_error": float(np.max(abs(values - reference))),
                }
            )
        )
    for workers in args.workers:
        controller = SimulatorController(workers=workers, batch_size=args.batch_size)
        times = []
        controller.connect()
        try:
            for _ in range(args.repeats + 1):
                start = perf_counter()
                values = controller.play(configs, [sequence], options, [[sweeper]])[
                    acq.id
                ]
                times.append(perf_counter() - start)
                error = float(np.max(abs(values - reference)))
                if error > 1e-5:
                    raise RuntimeError(f"Analytic accuracy requirement failed: {error}")
        finally:
            controller.disconnect()
        print(
            json.dumps(
                {
                    "workers": workers,
                    "cold_seconds": times[0],
                    "warm_median_seconds": float(np.median(times[1:])),
                    "max_absolute_error": error,
                }
            )
        )


if __name__ == "__main__":
    main()
