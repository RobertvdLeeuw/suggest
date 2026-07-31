async def with_backoff(
    call: Callable[[], Awaitable[T]],
    classify: Classify,
    waiting_time: float = 0.0,
    n_try: int = 1,
    max_tries: int = 5,
) -> T:
    if waiting_time:
        await asyncio.sleep(waiting_time)

    try:
        return await call()
    except Exception as exc:
        if n_try >= max_tries:
            raise

        match classify(exc):
            case "retry" | ("retry", wait_s):
                try:
                    next_wait = wait_s
                except NameError:
                    next_wait = max(waiting_time * 2, 1)

                logger.warning(
                    f"Attempt {n_try}/{max_tries} failed ({exc!r}), retrying in {next_wait:.2f}s"
                )
                return await with_backoff(call, classify, next_wait, n_try + 1, max_tries, min_wait)
            case "fatal":
                raise


def with_retry(classify: Classify, max_tries: int = 5):
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            return await with_backoff(lambda: fn(*args, **kwargs), classify, max_tries=max_tries)

        return wrapper

    return decorator
