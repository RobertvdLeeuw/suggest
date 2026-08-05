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
"""

# SongQueueDouble: same public surface as embedders.song_queue.SongQueue
# (put, get, remove, __contains__, __len__, peek, peek_all) - backed by
# collections.deque + threading.Lock + threading.Condition instead of
# multiprocessing.Manager().list() + mp.Lock + mp.Condition.
#
# needs:
#   - put(item): dedup semantics identical to the real class (no-op + return
#     False if item already present, else append + notify + return True).
#   - get(): blocks on empty via condition.wait(timeout=30), raises
#     TimeoutError on timeout, FIFO pop otherwise - match the real class's
#     timeout value exactly, it's a behavior contract, not an implementation
#     detail.
#   - remove(item): no-op if item not present, real removal otherwise.
#   - peek(): non-blocking-with-retry read of the head item, doesn't mutate.
#     (Real class currently polls with time.sleep(1) in a loop when empty -
#     decide whether the double should replicate that polling or just block
#     on the condition instead; note the choice, don't silently change the
#     contract being tested.)
#   - peek_all(): full snapshot list, doesn't mutate.
#   - __contains__ / __len__: same semantics as the real class.
#
# Deliberately NOT included: debug_status()/the lock-holder tracking fields -
# those are real-class debugging aids, not part of the tested contract.
#
# touches: collecter.embedders.song_queue.SongQueue (contract reference only,
#          this file must not import it - keep this a standalone double)
