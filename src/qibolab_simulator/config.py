"""Serializable physical parameters, with explicit channel-to-subsystem mapping."""

from typing import Literal, Self

import numpy as np
from pydantic import ConfigDict, Field, model_validator
from qibolab import AcquisitionConfig, Config, ConfigKinds
from qibolab._core.identifier import QubitId


class _Config(Config):
    model_config = ConfigDict(allow_inf_nan=False)


class QubitConfig(_Config):
    """Truncated Duffing oscillator; frequencies in Hz and coherence times in ns."""

    frequency: float = Field(gt=0)
    """Fundamental transition frequency, at zero flux bias."""
    anharmonicity: float = 0
    """Second transition frequency minus the first, negative for transmons."""
    levels: int = Field(default=2, ge=2)
    """Number of levels of the truncated oscillator."""
    t1: float | None = Field(default=None, gt=0)
    """Relaxation time of the fundamental transition, or no relaxation."""
    t2: float | None = Field(default=None, gt=0)
    """Coherence time of the fundamental transition, at most ``2 * t1``."""
    sweetspot: float = 0
    """Flux of maximum frequency, in flux quanta."""
    asymmetry: float = Field(default=0, ge=0, le=1)
    """Junction asymmetry of the split transmon."""

    @model_validator(mode="after")
    def coherence(self) -> Self:
        if self.t1 is not None and self.t2 is not None and self.t2 > 2 * self.t1:
            raise ValueError("t2 must not exceed 2*t1")
        return self

    def biased_frequency(self, flux: float | np.ndarray) -> float | np.ndarray:
        """Split-transmon frequency; flux is in units of the flux quantum."""
        factor = (
            self.asymmetry**2
            + (1 - self.asymmetry**2) * np.cos(np.pi * (flux - self.sweetspot)) ** 2
        ) ** 0.25
        return (self.frequency - self.anharmonicity) * factor + self.anharmonicity


class CouplingConfig(_Config):
    """Excitation-conserving exchange coupling, in Hz."""

    qubits: tuple[QubitId, QubitId]
    """The two coupled qubits."""
    coupling: float
    """Exchange strength."""


class HamiltonianConfig(_Config):
    """The simulated model, under the ``"hamiltonian"`` configuration key."""

    kind: Literal["simulator-hamiltonian"] = "simulator-hamiltonian"
    qubits: dict[QubitId, QubitConfig] = Field(min_length=1)
    """Subsystems, in order: each coupled component's frame is its first qubit's."""
    couplings: tuple[CouplingConfig, ...] = ()
    """Exchange couplings, at most one per pair."""

    @model_validator(mode="after")
    def valid_pairs(self) -> Self:
        seen = set()
        for pair in self.couplings:
            a, b = pair.qubits
            if a == b or a not in self.qubits or b not in self.qubits:
                raise ValueError(f"Invalid coupling targets: {pair.qubits}")
            key = frozenset(pair.qubits)
            if key in seen:
                raise ValueError(f"Duplicate coupling: {pair.qubits}")
            seen.add(key)
        return self


class DriveConfig(_Config):
    """Drive channel, in Hz."""

    kind: Literal["simulator-drive"] = "simulator-drive"
    target: QubitId
    """Driven qubit."""
    frequency: float = Field(gt=0)
    """Carrier frequency."""
    rabi_frequency: float = Field(default=1e9, gt=0)
    """Rabi frequency of a resonant, full-amplitude rectangular pulse."""


class FluxConfig(_Config):
    """Flux channel, biasing a split transmon; at most one per qubit."""

    kind: Literal["simulator-flux"] = "simulator-flux"
    target: QubitId
    """Biased qubit."""
    offset: float = 0
    """Static bias, to which flux pulse amplitudes add."""
    voltage_to_flux: float = 1
    """Conversion of offsets and amplitudes to flux quanta."""


class ReadoutConfig(AcquisitionConfig):
    """Ideal projective readout, not a resonator or electronics model."""

    model_config = ConfigDict(allow_inf_nan=False)
    kind: Literal["simulator-readout"] = "simulator-readout"
    target: QubitId
    """Measured qubit."""
    delay: float = 0
    """Must be zero: acquisition delays are not modeled."""
    smearing: float = 0
    """Must be zero: acquisition smearing is not modeled."""


def register_configs() -> None:
    """Register simulator runcard kinds with Qibolab (idempotently)."""
    kinds = (HamiltonianConfig, DriveConfig, FluxConfig, ReadoutConfig)
    registered = ConfigKinds.registered()
    ConfigKinds.extend(kind for kind in kinds if kind not in registered)
