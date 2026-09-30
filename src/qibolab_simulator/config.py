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
    anharmonicity: float = 0
    levels: int = Field(default=2, ge=2)
    t1: float | None = Field(default=None, gt=0)
    t2: float | None = Field(default=None, gt=0)
    sweetspot: float = 0
    asymmetry: float = Field(default=0, ge=0, le=1)

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
    coupling: float


class HamiltonianConfig(_Config):
    kind: Literal["simulator-hamiltonian"] = "simulator-hamiltonian"
    qubits: dict[QubitId, QubitConfig] = Field(min_length=1)
    couplings: tuple[CouplingConfig, ...] = ()

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
    kind: Literal["simulator-drive"] = "simulator-drive"
    target: QubitId
    frequency: float = Field(gt=0)
    rabi_frequency: float = Field(default=1e9, gt=0)


class FluxConfig(_Config):
    kind: Literal["simulator-flux"] = "simulator-flux"
    target: QubitId
    offset: float = 0
    voltage_to_flux: float = 1


class ReadoutConfig(AcquisitionConfig):
    """Ideal projective readout, not a resonator or electronics model."""

    model_config = ConfigDict(allow_inf_nan=False)
    kind: Literal["simulator-readout"] = "simulator-readout"
    target: QubitId
    delay: float = 0
    smearing: float = 0


def register_configs() -> None:
    """Register simulator runcard kinds with Qibolab (idempotently)."""
    kinds = (HamiltonianConfig, DriveConfig, FluxConfig, ReadoutConfig)
    registered = ConfigKinds.registered()
    ConfigKinds.extend(kind for kind in kinds if kind not in registered)
