from concurrent.futures import Future

import pytest

from qibolab_simulator.execution import batches, bounded_map, grouped_batches


class Executor:
    def __init__(self):
        self.futures = []

    def submit(self, function, item):
        future = Future()
        future.set_result(function(item))
        self.futures.append(future)
        return future


def test_bounded_submission():
    executor = Executor()
    iterator = bounded_map(executor, lambda x: x * 2, range(50), limit=3)
    first = next(iterator)
    assert len(executor.futures) == 3
    second = next(iterator)
    assert len(executor.futures) == 4
    assert sorted([first, second, *iterator]) == list(range(0, 100, 2))


def test_batches_and_validation():
    assert list(batches(range(5), 2)) == [(0, 1), (2, 3), (4,)]
    with pytest.raises(ValueError):
        list(batches([], 0))
    with pytest.raises(ValueError):
        list(bounded_map(Executor(), str, [], 0))


def test_failure_cancels_pending_tasks():
    class FailingExecutor:
        def __init__(self):
            self.futures = []

        def submit(self, function, item):
            future = Future()
            if item == 0:
                future.set_exception(RuntimeError("solver failed"))
            self.futures.append(future)
            return future

    executor = FailingExecutor()
    with pytest.raises(RuntimeError, match="solver failed"):
        list(bounded_map(executor, str, range(10), 3))
    assert len(executor.futures) == 3
    assert all(future.cancelled() for future in executor.futures[1:])


def test_slow_first_task_does_not_block_completed_results():
    class SlowExecutor(Executor):
        def submit(self, function, item):
            future = Future()
            if item != 0:
                future.set_result(item)
            self.futures.append(future)
            return future

    executor = SlowExecutor()
    iterator = bounded_map(executor, str, range(5), 2)
    assert next(iterator) == 1
    iterator.close()
    assert executor.futures[0].cancelled()


def test_grouped_batches():
    items = [(i % 3, i) for i in range(12)]
    grouped = list(grouped_batches(items, 4, key=lambda x: x[0], window=6))
    # Each window of six is regrouped, without crossing into the next one.
    assert [[k for k, _ in b] for b in grouped] == [
        [0, 0, 1, 1],
        [2, 2],
        [0, 0, 1, 1],
        [2, 2],
    ]
    assert sorted((i for b in grouped for i in b), key=lambda x: x[1]) == items
    with pytest.raises(ValueError):
        list(grouped_batches(items, 4, key=lambda x: x, window=0))
