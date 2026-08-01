"""
Strategies for clients/retry.py's generic with_backoff/with_retry engine -
deliberately client-agnostic, drives the engine with synthetic
exceptions/classify functions rather than any real Spotify/MusicBrainz/
LastFM exception type, since the engine itself doesn't know or care which
client it's retrying for.
"""

# classify_outcome_strat: one of "retry" / ("retry", wait_s) / "fatal" / an
# unrecognized value (to exercise with_backoff's ValueError-on-unexpected-
# classify-result branch) - needs all four represented, not just the two
# documented ones.
# touches: collecter.clients.retry.with_backoff (for reference on the match
#          cases it handles)

# failure_then_success_strat: a sequence of exceptions (length 0..N) to raise
# on successive calls before finally succeeding - drives with_backoff's
# actual retry loop across a realistic range of "fails a few times, then
# works" and "always fails" cases, paired with a classify function (from
# classify_outcome_strat) that's consistent across the whole sequence.
# needs a max_tries value drawn alongside it, so both "succeeds within
# budget" and "exhausts budget and raises" get covered.

# always_fatal_strat: an exception + a classify function that always returns
# "fatal" - specifically for the "fatal is never retried, single attempt
# only" assertion, kept separate from failure_then_success_strat since that
# one assumes eventual success.

# wait_time_strat: arbitrary non-negative floats for the explicit
# ("retry", wait_s) case, including 0.0 and very small/large values - for
# the "explicit wait_s is honored over the default doubling schedule"
# assertion.
