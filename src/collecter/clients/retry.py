import asyncio
import functools
import logging
from typing import Awaitable, Callable, TypeVar

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")
Classify = Callable[[Exception], "tuple[str, float] | str"]


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
            case "retry":
                next_wait = max(waiting_time * 2, 1)
            case ("retry", wait_s):
                next_wait = wait_s
            case "fatal":
                raise
            case other:
                # Unrecognized classify() result - treat as fatal rather than
                # silently falling out of the match with an implicit None return.
                raise ValueError(f"classify() returned unexpected value: {other!r}") from exc

        LOGGER.warning(
            f"Attempt {n_try}/{max_tries} failed ({exc!r}), retrying in {next_wait:.2f}s"
        )
        return await with_backoff(call, classify, next_wait, n_try + 1, max_tries)


def with_retry(classify: Classify, max_tries: int = 5):
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            return await with_backoff(lambda: fn(*args, **kwargs), classify, max_tries=max_tries)

        return wrapper

    return decorator
