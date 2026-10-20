"""Qibolab driver and terminal measurement processing."""

import multiprocessing
from collections.abc import Iterator
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
from pydantic import Field, PrivateAttr, model_validator
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Config,
    ExecutionParameters,
    PulseSequence,
)
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import PulseId
from qibolab._core.sweeper import ParallelSweepers
from threadpoolctl import threadpool_limits

from .engine import DynamiqsSettings, Settings, evolve
from .execution import batches, bounded_map
from .sequence import Experiment, compile_sequence
from .sweep import points, sweep_shape


@dataclass(frozen=True)
class _Job:
    coordinate: tuple[int, ...]
    sequence_index: int
    experiment: Experiment


def _measure(
    probabilities: np.ndarray,
    experiment: Experiment,
    options: ExecutionParameters,
    rng: np.random.Generator,
) -> dict[PulseId, np.ndarray]:
    dims = tuple(q.levels for q in experiment.model.qubits)
    singleshot = options.averaging_mode is AveragingMode.SINGLESHOT
    results = {}
    if singleshot:
        # One joint draw per shot preserves all simultaneous readout correlations.
        drawn = rng.choice(len(probabilities), size=options.nshots, p=probabilities)
        levels = np.unravel_index(drawn, dims)
    for identifier, target in experiment.acquisitions:
        if singleshot:
            value = (levels[target] != 0).astype(np.int8)
        else:
            ground = np.take(probabilities.reshape(dims), 0, axis=target).sum()
            value = np.asarray(np.clip(1 - ground, 0, 1))
        if options.acquisition_type is AcquisitionType.INTEGRATION:
            value = value.astype(float)
            value = np.stack((value, np.zeros_like(value)), axis=-1)
        results[identifier] = value
    return results


def _solve(
    jobs: tuple[_Job, ...],
    settings: Settings,
    backend: str,
    dynamiqs: DynamiqsSettings,
) -> list[np.ndarray]:
    problems = [
        (
            job.experiment.model,
            job.experiment.drives,
            job.experiment.fluxes,
            job.experiment.duration,
        )
        for job in jobs
    ]
    if backend == "qutip":
        return [evolve(*problem, settings) for problem in problems]
    # Imported on demand: dynamiqs is an optional dependency.
    from .dynamiqs_engine import evolve_batch

    return evolve_batch(problems, settings, dynamiqs)


def _run_batch(
    jobs: tuple[_Job, ...],
    *,
    settings: Settings,
    backend: str,
    dynamiqs: DynamiqsSettings,
    options: ExecutionParameters,
    seed: int,
    blas_threads: int,
) -> list[tuple[tuple[int, ...], dict[PulseId, np.ndarray]]]:
    results = []
    with threadpool_limits(limits=blas_threads):
        solved = _solve(jobs, settings, backend, dynamiqs)
        for job, probabilities in zip(jobs, solved, strict=True):
            exp = job.experiment
            # Coordinates, not worker numbers or completion order, define randomness.
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    seed, spawn_key=(job.sequence_index, *job.coordinate)
                )
            )
            results.append((job.coordinate, _measure(probabilities, exp, options, rng)))
    return results


class SimulatorController(Controller):
    """Rotating-frame controller; one worker by default.

    ``backend`` selects QuTiP or dynamiqs, which batches each submitted batch of
    experiments on a CPU, GPU, or TPU. Both share every other setting.
    ``executor`` can be a caller-owned Dask ``Client.get_executor()``.
    Use process workers with one execution thread each, not shared-thread solvers.
    """

    address: str = "local"
    sampling_rate_: float = Field(default=1, gt=0, allow_inf_nan=False)
    workers: int = Field(default=1, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    max_in_flight: int | None = Field(default=None, ge=1)
    blas_threads: int = Field(default=1, ge=1)
    seed: int | None = Field(default=None, ge=0)
    solver: Settings = Field(default_factory=Settings)
    backend: Literal["qutip", "dynamiqs"] = "qutip"
    dynamiqs: DynamiqsSettings = Field(default_factory=DynamiqsSettings)
    executor: Executor | None = Field(default=None, exclude=True, repr=False)
    _pool: ProcessPoolExecutor | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def process_executor(self):
        if isinstance(self.executor, ThreadPoolExecutor):
            raise ValueError(
                "Use a process executor: QuTiP integrators are not thread-safe"
            )
        if self.executor is not None and self.workers != 1:
            raise ValueError("Set workers=1 when supplying an external executor")
        return self

    @property
    def sampling_rate(self) -> float:
        return self.sampling_rate_

    @property
    def effective_batch_size(self) -> int:
        """Experiments per task: few for QuTiP, a whole sweep chunk for dynamiqs."""
        if self.batch_size is not None:
            return self.batch_size
        return 8 if self.backend == "qutip" else 1024

    def connect(self) -> None:
        """Keep a local process pool warm until disconnect."""
        if self.executor is None and self.workers > 1 and self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
            )

    def disconnect(self) -> None:
        """Release owned workers, never a caller-owned executor."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[PulseId, np.ndarray]:
        if options.acquisition_type is AcquisitionType.RAW:
            raise NotImplementedError(
                "RAW acquisition requires a resonator/readout model"
            )
        if options.fast_reset:
            raise NotImplementedError("Active reset is not modeled")
        if options.averaging_mode is AveragingMode.SINGLESHOT and (
            options.nshots is None or options.nshots < 1
        ):
            raise ValueError("SINGLESHOT requires a positive nshots")
        shape = sweep_shape(sweepers)
        identifiers = [p.id for sequence in sequences for _, p in sequence.acquisitions]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Acquisition identifiers must be unique across sequences")
        if not sequences:
            return {}

        def jobs() -> Iterator[_Job]:
            for coordinate, index, sequence, updated_configs in points(
                sequences, configs, sweepers
            ):
                yield _Job(
                    coordinate,
                    index,
                    compile_sequence(sequence, updated_configs, self.sampling_rate),
                )

        seed = self.seed if self.seed is not None else np.random.SeedSequence().entropy
        run = partial(
            _run_batch,
            settings=self.solver,
            backend=self.backend,
            dynamiqs=self.dynamiqs,
            options=options,
            seed=seed,
            blas_threads=self.blas_threads,
        )
        owned = self.executor is None and self._pool is None and self.workers > 1
        if owned:
            self.connect()
        executor = self.executor if self.executor is not None else self._pool
        tasks = batches(jobs(), self.effective_batch_size)
        completed = (
            map(run, tasks)
            if executor is None
            else bounded_map(
                executor, run, tasks, self.max_in_flight or 2 * self.workers
            )
        )
        result = {}
        try:
            for batch in completed:
                for coordinate, values in batch:
                    for identifier, value in values.items():
                        shots = options.averaging_mode is AveragingMode.SINGLESHOT
                        if identifier not in result:
                            inner = (
                                (2,)
                                if options.acquisition_type
                                is AcquisitionType.INTEGRATION
                                else ()
                            )
                            leading = (options.nshots,) if shots else ()
                            result[identifier] = np.empty(
                                leading + shape + inner, dtype=value.dtype
                            )
                        selection = (slice(None), *coordinate) if shots else coordinate
                        result[identifier][selection] = value
        finally:
            if executor is not None:
                completed.close()
            if owned:
                self.disconnect()
        return result
