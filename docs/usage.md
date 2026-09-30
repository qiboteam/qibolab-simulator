# Usage and physical model

## Minimal platform

The driver uses Qibolab's ordinary `Platform.execute()` interface. Channel names
are arbitrary: each simulator channel config explicitly targets a `QubitId`
(integer or string), rather than inferring it from the channel name.

```python
from qibolab import (
    Acquisition,
    AcquisitionChannel,
    AveragingMode,
    IqChannel,
    Parameters,
    Platform,
    Pulse,
    PulseSequence,
    Qubit,
    Rectangular,
)
from qibolab_simulator import (
    DriveConfig,
    HamiltonianConfig,
    QubitConfig,
    ReadoutConfig,
    SimulatorController,
    register_configs,
)

register_configs()  # Also call before loading serialized simulator configs.
controller = SimulatorController(
    seed=1234,
    channels={"xy": IqChannel(), "adc": AcquisitionChannel()},
)
platform = Platform(
    name="one-qubit",
    instruments={"simulator": controller},
    qubits={"q": Qubit(drive="xy", acquisition="adc")},
    parameters=Parameters(
        configs={
            "hamiltonian": HamiltonianConfig(
                qubits={"q": QubitConfig(frequency=5e9, levels=2)}
            ),
            "xy": DriveConfig(target="q", frequency=5e9, rabi_frequency=12.5e6),
            "adc": ReadoutConfig(target="q"),
        }
    ),
)
pulse = Pulse(duration=40, amplitude=1, envelope=Rectangular())
acquisition = Acquisition(duration=100)
sequence = PulseSequence([("xy", pulse)])
sequence.align(["xy", "adc"])
sequence.append(("adc", acquisition))

# Put execution inside `if __name__ == "__main__":` in scripts using local
# multiprocessing. This one-worker example runs in the calling process.
platform.connect()
try:
    result = platform.execute([sequence], averaging_mode=AveragingMode.CYCLIC)
    print(result[acquisition.id])  # Approximately 1: a resonant pi pulse.
finally:
    platform.disconnect()
```

For a full amplitude scan, see [examples/rabi.py](../examples/rabi.py).
Execution results are keyed by acquisition identifiers, which must be unique
across submitted sequences.

## Units and approximations

- Frequencies, anharmonicities, exchange couplings, and `rabi_frequency`: **Hz**.
- Durations, `t1`, and `t2`: **ns**; phases: **radians**.
- `sampling_rate_`: samples/ns (default `1`); pulse durations must fit this grid.
- `FluxConfig.offset` and pulse amplitude are converted to flux-quantum units
  with `voltage_to_flux`. `QubitConfig.sweetspot` is in flux-quantum units.

Each subsystem is a truncated Duffing oscillator with its own `levels`.
`CouplingConfig(qubits=("a", "b"), coupling=...)` adds excitation-conserving
exchange coupling. Each exchange-connected component uses one shared rotating
frame, selected from its first qubit's frequency in configured subsystem order.
This removes a common carrier, not all detunings or all potentially fast dynamics.

With amplitude-scaled envelope samples `I(t), Q(t)`, the coefficient multiplying
the creation operator follows

```text
-i*pi*rabi_frequency/1e9 * (I(t) + i*Q(t))
    * exp[-i*(2*pi*(drive_frequency - frame_frequency)*t/1e9 + phase)]
```

The phase includes the pulse's `relative_phase` and accumulated channel
`VirtualZ` phases. The rotating-wave approximation drops drive counterrotating
terms; this is **not** an exact change of coordinates for the full lab-frame
driven model. Validate strong-drive and large-detuning regimes before relying on
the approximation.

Coherence parameters are optional scalar fundamental-transition `t1`/`t2`
values, not independently configurable rates for every transition. When both
are supplied, `t2 <= 2*t1` is required. Flux channels use a split-transmon
frequency-versus-flux model; only one flux channel per subsystem is supported.
Flux envelopes must be real, without a pulse phase.

## Acquisition contract

Every realized experiment starts in the joint ground state. All acquisitions
must start simultaneously after control has finished. Acquisition duration does
not introduce a resonator or electronics simulation.

| Mode | Returned value per acquisition |
| --- | --- |
| Averaged discrimination | Exact `p(non-ground)` |
| Single-shot discrimination | Binary non-ground indicator for each shot |
| Averaged integration | Ideal IQ pair `(p(non-ground), 0)` |
| Single-shot integration | Ideal IQ pair `(binary outcome, 0)` per shot |

Single-shot sampling draws the **joint system state once per shot**, retaining
correlations between simultaneous acquisitions. Levels above zero are all
classified as `1`: simulating qutrit leakage is not three-outcome discrimination.
There is no confusion matrix, Gaussian IQ noise, or resonator response.

For sweep shape `S`, discrimination results have shape `S` when averaged and
`(nshots, *S)` in single-shot mode; integration adds a final axis of length two.
Use `AcquisitionType.INTEGRATION` for IQ output and
`AveragingMode.SINGLESHOT` with a positive `nshots` for shots.

RAW acquisition, mid-circuit/non-simultaneous acquisitions, active reset,
chirped pulses, and nonzero readout delay/smearing/time-of-flight are unsupported
and rejected explicitly. The ground-state initialization is not an active-reset
simulation or a model of memory between shots.

## Sweeps and timing

Pass sweepers as `[[sweeper_a, sweeper_b], [sweeper_c]]`: members of one inner list
vary together and must have equal nonzero lengths; outer lists form Cartesian
axes. Supported targets include pulse amplitude, relative phase, duration, and
`duration_interpolated`, drive-channel frequency, and flux-channel offset.

Sequence timing and `Align` instructions are resolved **after each point's
updates**, so delay/duration sweeps move acquisitions correctly. The waveform
sampling grid remains fixed. `duration_interpolated` maps to duration and
resamples the envelope on that same grid; it does not emulate a hardware
interpolation mechanism. Off-grid control-pulse durations are rejected.

## Migration and deliberate gaps

Import from `qibolab_simulator` and register its config kinds:
`simulator-hamiltonian`, `simulator-drive`, `simulator-flux`, and
`simulator-readout`. Supply the Hamiltonian under the `"hamiltonian"` config key
and give every channel an explicit `target`. Port old runcards deliberately;
the old emulator schema/imports are not compatibility aliases.

Move level counts into individual `QubitConfig` objects. Convert old
per-transition T1/T2 arrays to scalar parameters **only if the intended physics
is the fundamental-transition damping model**; otherwise this model is not an
equivalent replacement.

The upstream roadmap motivates this package but does not promise feature parity:

- [Independent packaging (#1257)](https://github.com/qiboteam/qibolab/issues/1257)
  and [rotating frames (#1453)](https://github.com/qiboteam/qibolab/issues/1453):
  the main direction of this driver.
- [Collapsed measurements (#1263)](https://github.com/qiboteam/qibolab/issues/1263):
  no mid-circuit collapse here; only terminal measurement.
- [RAW traces (#1173)](https://github.com/qiboteam/qibolab/issues/1173) and
  [confusion-matrix processing (#1413)](https://github.com/qiboteam/qibolab/issues/1413):
  no readout-device or assignment-error model.
- [External operators (#1256)](https://github.com/qiboteam/qibolab/issues/1256)
  and [diagnostics (#1409)](https://github.com/qiboteam/qibolab/issues/1409):
  arbitrary operator providers and Hamiltonian/state-trace dump APIs are not
  provided by this execution interface.
