"""Tests for the bounded, in-order pool map that ``prepare`` standardises through.

``prepare --subset 10k`` used to hang just after its progress bar filled: it
broke out of a ``pool.imap`` that had queued all of ChEMBL and then called
``Pool.terminate()`` on a busy pool, which can deadlock. These pin the two
properties the replacement must have -- results in input order, so shards stay
byte-identical, and a bounded number of items in flight, so stopping early
leaves a pool that ``close()`` and ``join()`` can shut down.
"""

from __future__ import annotations

import multiprocessing
import operator
import threading

import pytest

from morg2smiles.data.prepare import _imap_bounded


class _Done:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _RecordingPool:
    """Runs submissions inline and records how many have been made."""

    def __init__(self):
        self.submitted = 0

    def apply_async(self, func, args):
        self.submitted += 1
        return _Done(func(*args))


def test_results_come_back_in_input_order():
    with multiprocessing.Pool(3) as pool:
        out = list(_imap_bounded(pool, operator.neg, range(200), window=4))
        pool.close()
        pool.join()
    assert out == [-x for x in range(200)]


def test_never_more_than_window_items_ahead_of_the_consumer():
    pool, window, n = _RecordingPool(), 5, 40
    for consumed, _ in enumerate(_imap_bounded(pool, operator.neg, range(n), window), start=1):
        assert pool.submitted == min(n, window + consumed)
        if consumed == 7:
            break
    # Stopping early leaves only the window outstanding, not the whole input.
    assert pool.submitted == window + 7


def test_empty_input_yields_nothing():
    assert list(_imap_bounded(_RecordingPool(), operator.neg, [], window=3)) == []


def test_window_must_be_positive():
    with pytest.raises(ValueError, match="window"):
        next(_imap_bounded(_RecordingPool(), operator.neg, [1], window=0))


def test_stopping_early_leaves_a_pool_that_shuts_down():
    """The hang itself: results larger than a pipe buffer, abandoned mid-stream.

    ``bytes(200_000)`` stands in for a 2,000-molecule chunk's result, which is
    larger than the pipe it travels back through. Run in a thread so a
    regression fails this test instead of hanging the suite.
    """
    finished = threading.Event()

    def run():
        with multiprocessing.Pool(4) as pool:
            for i, _ in enumerate(_imap_bounded(pool, bytes, [200_000] * 300, window=8)):
                if i == 5:
                    break
            pool.close()
            pool.join()
        finished.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=120)
    assert finished.is_set(), "pool did not shut down after stopping early"
