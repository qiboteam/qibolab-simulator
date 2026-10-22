"""Shared fixtures, and the selection of slow tests.

Tests marked ``slow`` run in full by default. ``--slow`` runs ``all``, ``none``,
or a random fraction of them, drawn per test from ``--slow-seed``. Pushes to a
branch in GitHub Actions default to a quarter, seeded by the commit hash, so
that successive commits exercise different slow tests.
"""

import os
import secrets
from hashlib import sha256

import pytest
from qibolab import Acquisition, Pulse, PulseSequence, Rectangular

from qibolab_simulator import DriveConfig, HamiltonianConfig, QubitConfig, ReadoutConfig


@pytest.fixture
def configs():
    return {
        "hamiltonian": HamiltonianConfig(qubits={"alice": QubitConfig(frequency=5e9)}),
        "microwave": DriveConfig(target="alice", frequency=5e9, rabi_frequency=12.5e6),
        "detector": ReadoutConfig(target="alice"),
    }


@pytest.fixture
def experiment():
    pulse = Pulse(duration=40, amplitude=1, envelope=Rectangular())
    readout = Acquisition(duration=100)
    sequence = PulseSequence([("microwave", pulse)])
    sequence.align(["microwave", "detector"])
    sequence.append(("detector", readout))
    return sequence, pulse, readout


# Shared with xdist workers, which inherit the controller's environment.
_SEED = "QIBOLAB_SIMULATOR_SLOW_SEED"
_PUSH_FRACTION = "0.25"


def pytest_addoption(parser):
    parser.addoption(
        "--slow",
        default=None,
        help='Slow tests to run: "all", "none", or a fraction (default: all, or '
        f"{_PUSH_FRACTION} on GitHub Actions branch pushes)",
    )
    parser.addoption("--slow-seed", default=None, help="Seed of the slow-test sample")


def _push() -> bool:
    return os.environ.get("GITHUB_EVENT_NAME") == "push" and not os.environ.get(
        "GITHUB_REF", ""
    ).startswith("refs/tags/")


def _fraction(config) -> float:
    option = config.getoption("--slow")
    if option is None:
        option = _PUSH_FRACTION if _push() else "all"
    fraction = {"all": 1.0, "none": 0.0}.get(option)
    if fraction is None:
        try:
            fraction = float(option)
        except ValueError:
            fraction = -1.0
        if not 0 <= fraction <= 1:
            raise pytest.UsageError('--slow must be "all", "none", or within [0, 1]')
    return fraction


def _seed(config) -> str | None:
    seed = config.getoption("--slow-seed") or os.environ.get(_SEED)
    if seed is None and _push():
        seed = os.environ.get("GITHUB_SHA")
    return seed


def pytest_configure(config):
    # Draw a seed once, before xdist starts its workers.
    if not hasattr(config, "workerinput") and _seed(config) is None:
        os.environ[_SEED] = secrets.token_hex(4)


def pytest_xdist_auto_num_workers(config):
    """Count the usable CPUs: xdist's psutil count ignores affinity and cgroups."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return None


def pytest_report_header(config):
    fraction = _fraction(config)
    if fraction in (0, 1):
        return f"slow tests: {'all' if fraction else 'none'}"
    seed = _seed(config)
    return f"slow tests: {fraction:.0%} sample, rerun with --slow-seed={seed}"


def pytest_collection_modifyitems(config, items):
    fraction = _fraction(config)
    if fraction == 1:
        return
    seed = _seed(config)

    def drawn(item) -> bool:
        digest = sha256(f"{seed}:{item.nodeid}".encode()).digest()
        return int.from_bytes(digest[:8]) / 2**64 < fraction

    deselected = [i for i in items if "slow" in i.keywords and not drawn(i)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        dropped = set(map(id, deselected))
        items[:] = [i for i in items if id(i) not in dropped]
