import numpy as np
import pytest
from qibolab import (
    Acquisition,
    Delay,
    Pulse,
    PulseSequence,
    Readout,
    Rectangular,
    VirtualZ,
)

from qibolab_simulator import FluxConfig
from qibolab_simulator.sequence import compile_sequence


def test_exact_zero_time(configs):
    result = compile_sequence(
        PulseSequence([("detector", Acquisition(duration=20))]), configs, 1
    )
    assert result.duration == 0
    assert not result.drives


def test_drive_phase_and_virtual_z(configs, experiment):
    sequence, pulse, _ = experiment
    sequence.insert(0, ("microwave", VirtualZ(phase=np.pi / 2)))
    compiled = compile_sequence(sequence, configs, 1)
    expected = -np.pi * configs["microwave"].rabi_frequency / 1e9
    np.testing.assert_allclose(compiled.drives[0].samples, expected, atol=1e-17)
    assert pulse.relative_phase == 0


def test_flux_bias_and_pulse(configs, experiment):
    configs["bias"] = FluxConfig(target="alice", offset=0.1, voltage_to_flux=0.5)
    sequence, _, _ = experiment
    sequence.insert(
        0, ("bias", Pulse(duration=20, amplitude=0.1, envelope=Rectangular()))
    )
    compiled = compile_sequence(sequence, configs, 1)
    qubit = configs["hamiltonian"].qubits["alice"]
    assert compiled.model.qubits[0].frequency == qubit.biased_frequency(0.05)
    expected = (
        2 * np.pi * (qubit.biased_frequency(0.1) - qubit.biased_frequency(0.05)) / 1e9
    )
    np.testing.assert_allclose(compiled.fluxes[0].samples, expected)


@pytest.mark.parametrize("duration", [-1, np.nan, np.inf, 0.5])
def test_invalid_duration(configs, experiment, duration):
    sequence, pulse, _ = experiment
    sequence[0] = ("microwave", pulse.model_copy(update={"duration": duration}))
    with pytest.raises(ValueError):
        compile_sequence(sequence, configs, 1)


def test_no_measurements(configs, experiment):
    with pytest.raises(ValueError, match="terminal acquisition"):
        compile_sequence(PulseSequence([experiment[0][0]]), configs, 1)


def test_mid_circuit_measurements_fail(configs, experiment):
    sequence, _, _ = experiment
    sequence.append(("detector", Acquisition(duration=10)))
    with pytest.raises(NotImplementedError, match="simultaneous terminal"):
        compile_sequence(sequence, configs, 1)


def test_pulse_after_readout_fails(configs, experiment):
    sequence, pulse, _ = experiment
    sequence.append(("microwave", pulse.new()))
    with pytest.raises(NotImplementedError, match="beyond measurement"):
        compile_sequence(sequence, configs, 1)


def test_readout_wrapper_preserves_acquisition_id(configs):
    readout = Readout.from_probe(
        Pulse(duration=20, amplitude=0.1, envelope=Rectangular())
    )
    compiled = compile_sequence(PulseSequence([("detector", readout)]), configs, 1)
    assert compiled.acquisitions[0][0] == readout.acquisition.id


def test_fractional_delay_is_not_sampled(configs):
    sequence = PulseSequence(
        [("detector", Delay(duration=0.125)), ("detector", Acquisition(duration=1))]
    )
    assert compile_sequence(sequence, configs, 1).duration == 0.125
