"""
SongQueue's logical contract (dedup on put, FIFO get, peek doesn't mutate,
len stays accurate) tested as a single hypothesis.stateful.RuleBasedStateMachine,
parametrized over two backing implementations:

  - mocks.song_queue_double.SongQueueDouble - deque + threading primitives,
    no multiprocessing.Manager() subprocess spin-up. Run at hypothesis's
    normal/generous step budget - this is where volume of explored
    put/get/remove/peek interleavings actually finds bugs.
  - collecter.embedders.song_queue.SongQueue (the real class) - run at a
    small, capped budget. Not redundant with the double: catches anything
    that depends on going through Manager().list()'s proxying (e.g. dedup
    relying on equality vs identity across the proxy boundary) that a local
    deque can't surface. Kept small specifically because Manager() spin-up
    cost is real and hypothesis's value here is confirmatory, not exploratory.

Real cross-process guarantees (no deadlock, no lost items under actual
concurrent OS processes) are explicitly NOT covered here - see
integration/test_song_queue_concurrency.py.
"""

# SongQueueContract(RuleBasedStateMachine): shared rule set + invariants,
# instantiated once per backing implementation via a fixture/factory bundle
# rather than duplicated per-class.
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
# touches: collecter.embedders.song_queue.SongQueue, mocks.song_queue_double.SongQueueDouble,
#          strategies.queue.song_queue_item_strat, hypothesis.stateful.RuleBasedStateMachine

# test_song_queue_double_contract: runs SongQueueContract against
# SongQueueDouble at normal hypothesis settings (generous step count).

# test_song_queue_real_contract: runs SongQueueContract against the real
# SongQueue at a small capped step count, @pytest.mark.slow (still faster
# than the double's full run, but not free given Manager() spin-up).
