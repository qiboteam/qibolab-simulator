"""Bounded scheduling shared by local processes and cluster executors."""

from collections.abc import Callable, Hashable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Executor, wait
from itertools import islice
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def batches(items: Iterable[T], size: int) -> Iterator[tuple[T, ...]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    iterator = iter(items)
    while batch := tuple(islice(iterator, size)):
        yield batch


def grouped_batches(
    items: Iterable[T], size: int, key: Callable[[T], Hashable], window: int
) -> Iterator[tuple[T, ...]]:
    """Batches of ``window`` items at a time, reordered so equal keys are adjacent."""
    if window < 1:
        raise ValueError("window must be positive")
    iterator = iter(items)
    while chunk := list(islice(iterator, window)):
        groups: dict[Hashable, list[T]] = {}
        for item in chunk:
            groups.setdefault(key(item), []).append(item)
        yield from batches((i for group in groups.values() for i in group), size)


def bounded_map(
    executor: Executor, function: Callable[[T], R], items: Iterable[T], limit: int
) -> Iterator[R]:
    """Consume at most ``limit`` pending inputs, yielding completed results."""
    if limit < 1:
        raise ValueError("pending task limit must be positive")
    iterator = iter(items)
    pending = set()
    try:
        for item in islice(iterator, limit):
            pending.add(executor.submit(function, item))
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.remove(future)
                yield future.result()
                for item in islice(iterator, 1):
                    pending.add(executor.submit(function, item))
    finally:
        for future in pending:
            future.cancel()
