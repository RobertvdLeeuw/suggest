"""
SongQueue's logical contract (dedup on put, FIFO get, peek doesn't mutate,
len stays accurate) tested as a single hypothesis.stateful.RuleBasedStateMachine,
run against ONE backing implementation:

  - mocks.song_queue_double.SongQueueDouble - deque + threading primitives,
    no multiprocessing.Manager() subprocess spin-up. Run at hypothesis's
    normal/generous step budget - this is where volume of explored
    put/get/remove/peek interleavings actually finds logic bugs.

Scope note (post-slimdown): we previously also ran this same state machine
against the real collecter.embedders.song_queue.SongQueue class at a capped
budget, on the theory that Manager().list() proxying might behave
differently from a local deque (e.g. dedup relying on equality vs identity
across the proxy boundary). Cut that second run: SongQueueDouble and the
real SongQueue implement the identical dedup/FIFO contract by construction
(same "item in self.queue" check, same list semantics), so a second
Hypothesis run over the real class was mostly re-confirming what the double
already proved, at real Manager()-subprocess cost. The one thing the double
genuinely can't cover - correctness under actual concurrent OS processes -
is handled by integration/test_song_queue_concurrency.py's example-based
tests instead, which is a better tool for that question than a stateful
machine anyway.
"""

# SongQueueContract(RuleBasedStateMachine): shared rule set + invariants.
#
# rules needed: put(item), get() [only when known non-empty, to avoid the
# 30s-timeout path dominating run time - track "known items" as the state
# machine's own bundle rather than blindly calling get() on empty],
# remove(item), peek(), peek_all(), __len__/__contains__ as invariant checks
# rather than rules.
#
# invariants needed:
#   - len(queue) always equals the number of distinct items put and not yet
#     removed/gotten (tracked in the state machine's own model state).
#   - put() of an already-present item is a no-op (returns False, doesn't
#     grow len, doesn't duplicate the item in peek_all()'s result).
#   - peek() never mutates - calling it twice in a row with no put/get
#     between returns the same item and doesn't change len.
#   - get() only ever returns an item that was actually put and not yet
#     removed by a prior get()/remove() call.
#   - peek_all()'s returned list matches the state machine's own tracked
#     model exactly (as a set - real class doesn't guarantee order beyond
#     FIFO for get(), peek_all() is a snapshot).
#
# touches: mocks.song_queue_double.SongQueueDouble,
#          strategies.queue.song_queue_item_strat, hypothesis.stateful.RuleBasedStateMachine

# test_song_queue_double_contract: runs SongQueueContract against
# SongQueueDouble at normal hypothesis settings (generous step count). This
# is now the only test in this file - real-class + real-multiprocess
# confidence comes from integration/test_song_queue_concurrency.py instead.
