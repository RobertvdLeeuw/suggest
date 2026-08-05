# API calls that fail due to networking (not 404 or similar) are handled with
# exponential backoff. (Client-agnostic version - the old test_api.py assertion was
# API-specific; the actual backoff *engine* lives in clients/retry.py and doesn't
# know which API it's retrying for, so it's tested here once instead of per-client.)
# touches: collecter.clients.retry.with_backoff, strategies.retry.failure_then_success_strat

# with_backoff never retries beyond max_tries, even if classify() always returns "retry".
# touches: collecter.clients.retry.with_backoff, strategies.retry.always_fatal_strat (inverted -
#          need an "always retry" variant, not just "always fatal")

# A "fatal" classification is never retried - exactly one call attempt.
# touches: collecter.clients.retry.with_backoff, strategies.retry.always_fatal_strat

# When classify() returns an explicit ("retry", wait_s), that wait is honored over the
# default doubling schedule (max(waiting_time * 2, 1)).
# touches: collecter.clients.retry.with_backoff, strategies.retry.wait_time_strat

# An unrecognized classify() return value raises ValueError rather than silently
# falling through (the match/case "other" branch).
# touches: collecter.clients.retry.with_backoff, strategies.retry.classify_outcome_strat

# with_retry's decorator wraps a function such that its own call signature/return value
# is preserved on success (functools.wraps contract) - a thin but worth-having check
# given every client wrapper method uses this decorator.
# touches: collecter.clients.retry.with_retry
