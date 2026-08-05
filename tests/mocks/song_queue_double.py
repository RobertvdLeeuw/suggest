"""
Lightweight in-process double for embedders.song_queue.SongQueue, used
alongside (not instead of) the real class in unit/test_song_queue.py's
stateful property tests - see that file's module docstring for the
real-vs-double split and why it's safe here specifically (SongQueue's
methods are already thin wrappers around lock/condition primitives, and
threading.Condition mirrors multiprocessing.Condition's API closely enough
that swapping the backing store is a low-risk substitution, unlike faking a
third-party API response shape).

Must NOT diverge from the real SongQueue's documented semantics. If a rule
in unit/test_song_queue.py's shared RuleBasedStateMachine fails only against
this double and not the real class (or vice versa), that's this file's bug,
not the state machine's.

Deliberate divergence, noted per the real class's docstring conventions:
peek() here blocks on the condition (wait, re-check, wait...) instead of
replicating the real class's poll-with-time.sleep(1) loop. Both eventually
return the head item once one exists and neither mutates the queue -
sleep-vs-wait is an implementation detail of *how* it blocks, not part of
the observable contract RuleBasedStateMachine's rules/invariants check, so
using the condition here (rather than a real 1s sleep loop, which would tax
Hypothesis's generous step budget) is not a semantics change - just a faster
implementation of the same contract.
"""

import threading
from collections import deque


class SongQueueDouble:
    """Same public surface as embedders.song_queue.SongQueue (put, get,
    remove, __contains__, __len__, peek, peek_all) - backed by
    collections.deque + threading.Lock + threading.Condition instead of
    multiprocessing.Manager().list() + mp.Lock + mp.Condition.

    Deliberately NOT included: debug_status()/the lock-holder tracking
    fields - those are real-class debugging aids, not part of the tested
    contract.
    """

    def __init__(self, name: str = "double"):
        self.name = name
        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def put(self, item) -> bool:
        with self._lock:
            if item in self._queue:
                return False
            self._queue.append(item)
            self._condition.notify()
            return True

    def get(self):
        with self._condition:
            while len(self._queue) == 0:
                if not self._condition.wait(timeout=30):  # same 30s contract as the real class
                    raise TimeoutError(f"Queue {self.name} timeout after 30s")
            return self._queue.popleft()

    def remove(self, item) -> None:
        with self._lock:
            try:
                self._queue.remove(item)
            except ValueError:
                pass  # not present - no-op, matches the real class

    def __contains__(self, item) -> bool:
        with self._lock:
            return item in self._queue

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    def peek(self):
        """Blocks on the condition until an item exists, then returns the
        head without removing it. See module docstring for why this blocks
        on the condition rather than the real class's sleep(1) poll loop."""
        with self._condition:
            while len(self._queue) == 0:
                self._condition.wait()
            return self._queue[0]

    def peek_all(self) -> list:
        with self._lock:
            return list(self._queue)
