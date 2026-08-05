"""
Strategies for clients/retry.py's generic with_backoff/with_retry engine -
deliberately client-agnostic, drives the engine with synthetic
exceptions/classify functions rather than any real Spotify/MusicBrainz/
LastFM exception type, since the engine itself doesn't know or care which
client it's retrying for.
"""

import hypothesis.strategies as st

# classify_outcome_strat: one of "retry" / ("retry", wait_s) / "fatal" / an
# unrecognized value - needs all four represented (not just the two
# documented match cases) so with_backoff's ValueError-on-unexpected-result
# branch actually gets exercised, not just the two happy-path cases.
classify_outcome_strat = st.one_of(
    st.just("retry"),
    st.tuples(st.just("retry"), st.floats(min_value=0.0, max_value=60.0, allow_nan=False)),
    st.just("fatal"),
    # unrecognized: anything that isn't one of the three shapes above -
    # covers both "wrong string" and "right-shaped-looking but wrong" cases.
    st.one_of(
        st.text(min_size=1).filter(lambda s: s not in ("retry", "fatal")),
        st.integers(),
        st.none(),
    ),
)

# wait_time_strat: arbitrary non-negative floats, including 0.0 and very
# small/large values - for the explicit ("retry", wait_s) case, where the
# assertion is that with_backoff honors wait_s exactly rather than falling
# back to its own doubling schedule.
wait_time_strat = st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)


class _ProbeError(Exception):
    """Synthetic exception raised by the fake `call()` under test - carries
    an index so a test can confirm exactly which attempt raised it, without
    needing a real client exception type."""

    def __init__(self, attempt: int):
        super().__init__(f"synthetic failure on attempt {attempt}")
        self.attempt = attempt


@st.composite
def failure_then_success_strat(draw) -> tuple[list[Exception], int]:
    """(exceptions, max_tries) where len(exceptions) < max_tries is
    guaranteed - eventual success is always reachable within budget. Pairs
    with a classify function that always returns "retry" for every
    exception in the list (retryable is the only case worth exercising here;
    the "fatal stops immediately" case is always_fatal_strat's job, and
    mixing the two into one strategy would make each generated case's
    postcondition ambiguous - was budget exhausted, or did a fatal cut it
    short? - which is exactly the ambiguity keeping these as two separate
    strategies avoids)."""
    max_tries = draw(st.integers(min_value=1, max_value=6))
    n_failures = draw(st.integers(min_value=0, max_value=max_tries - 1))
    exceptions = [_ProbeError(i) for i in range(n_failures)]
    return exceptions, max_tries


@st.composite
def exhausts_budget_strat(draw) -> tuple[list[Exception], int]:
    """(exceptions, max_tries) where len(exceptions) >= max_tries is
    guaranteed - every attempt within budget fails, so with_backoff must
    exhaust and re-raise the last exception. Every failure classifies as
    "retry" (never "fatal") - a fatal failure would exhaust the budget for
    the wrong reason (short-circuit, not attrition), which is specifically
    what always_fatal_strat tests instead."""
    max_tries = draw(st.integers(min_value=1, max_value=6))
    n_failures = draw(st.integers(min_value=max_tries, max_value=max_tries + 3))
    exceptions = [_ProbeError(i) for i in range(n_failures)]
    return exceptions, max_tries


@st.composite
def always_fatal_strat(draw) -> Exception:
    """A single exception, paired (by the test, via a fixed `lambda exc:
    "fatal"` classify function - not generated here, since "always fatal" IS
    the classify function, there's nothing to vary about it) for the "fatal
    is never retried, single attempt only" assertion. Kept separate from
    failure_then_success_strat/exhausts_budget_strat since those both assume
    a "retry" classification throughout."""
    return _ProbeError(draw(st.integers(min_value=0, max_value=10)))
