"""Lazy Cartesian sweeps with synchronized values within each axis."""

from collections.abc import Iterator
from itertools import product

import numpy as np
from qibolab import Config, Parameter, PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from .config import DriveConfig, FluxConfig


def sweep_shape(sweepers: list[ParallelSweepers]) -> tuple[int, ...]:
    shape = []
    for axis in sweepers:
        if not axis:
            raise ValueError("Sweep axes cannot be empty")
        lengths = {len(s.values) for s in axis}
        if len(lengths) != 1 or 0 in lengths:
            raise ValueError("Parallel sweepers must have equal nonzero lengths")
        for sweeper in axis:
            if sweeper.values.ndim != 1 or not np.all(np.isfinite(sweeper.values)):
                raise ValueError("Sweep values must be finite one-dimensional arrays")
        shape.append(lengths.pop())
    return tuple(shape)


def validate_targets(
    sequences: list[PulseSequence],
    configs: dict[str, Config],
    sweepers: list[ParallelSweepers],
) -> None:
    pulses = {p.id: p for sequence in sequences for _, p in sequence}
    seen = set()
    for axis in sweepers:
        for sweeper in axis:
            parameter = sweeper.parameter
            name = (
                "duration"
                if parameter is Parameter.duration_interpolated
                else parameter.name
            )
            targets = (
                [p.id for p in sweeper.pulses]
                if sweeper.pulses is not None
                else sweeper.channels
            )
            if not targets:
                raise ValueError("Sweepers must have at least one target")
            for target in targets:
                key = (target, name)
                if key in seen:
                    raise ValueError(f"Duplicate sweep target for {name}")
                seen.add(key)
                if sweeper.pulses is not None:
                    if (
                        target not in pulses
                        or name not in type(pulses[target]).model_fields
                    ):
                        raise ValueError(f"Invalid pulse sweep target for {name}")
                elif target not in configs or not (
                    (
                        parameter is Parameter.frequency
                        and isinstance(configs[target], DriveConfig)
                    )
                    or (
                        parameter is Parameter.offset
                        and isinstance(configs[target], FluxConfig)
                    )
                ):
                    raise ValueError(f"Invalid channel sweep target for {name}")


def points(
    sequences: list[PulseSequence],
    configs: dict[str, Config],
    sweepers: list[ParallelSweepers],
) -> Iterator[tuple[tuple[int, ...], int, PulseSequence, dict[str, Config]]]:
    """Yield updated sequences without mutating pulses, configs, or alignment."""
    shape = sweep_shape(sweepers)
    validate_targets(sequences, configs, sweepers)
    for coordinate in product(*(range(size) for size in shape)):
        pulse_updates, config_updates = {}, {}
        for axis, index in zip(sweepers, coordinate, strict=True):
            for sweeper in axis:
                name = (
                    "duration"
                    if sweeper.parameter is Parameter.duration_interpolated
                    else sweeper.parameter.name
                )
                value = float(sweeper.values[index])
                if sweeper.pulses is not None:
                    for pulse in sweeper.pulses:
                        pulse_updates.setdefault(pulse.id, {})[name] = value
                else:
                    for channel in sweeper.channels:
                        config_updates.setdefault(channel, {})[name] = value
        updated_configs = {
            key: type(config).model_validate(dict(config) | config_updates.get(key, {}))
            if key in config_updates
            else config
            for key, config in configs.items()
        }
        for sequence_index, sequence in enumerate(sequences):
            updated = PulseSequence(
                [
                    (channel, pulse.model_copy(update=pulse_updates[pulse.id]))
                    if pulse.id in pulse_updates
                    else (channel, pulse)
                    for channel, pulse in sequence
                ]
            )
            yield coordinate, sequence_index, updated, updated_configs
