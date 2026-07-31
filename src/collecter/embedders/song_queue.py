"""
Purpose & scope
---------------
SongQueue - the shared multiprocess queue jukemir.py and auditus.py's worker
processes pull from - plus the typing surface each embedder's embed function
must satisfy.

Two separate Protocols (JukeMIREmbedFunc, AuditusEmbedFunc), not one shared
one, because each embedder returns a different Embedding* type. __init__.py's
EMBEDDERS registry pairs each embed function with its matching queue/embedding
types, so keeping the return types distinct here lets that pairing be typed
instead of just asserted at runtime.

Does NOT know about jukemirlib, auditus, or how an embed function actually
produces its embeddings - it only describes the shape of the queue and the
shape a conforming embed function must have.
"""

import logging
import multiprocessing as mp
import os
import time
from typing import Protocol

from models import EmbeddingAuditus, EmbeddingJukeMIR, QueueAuditus, QueueJukeMIR

LOGGER = logging.getLogger(__name__)

QueueObject = QueueJukeMIR | QueueAuditus
QUEUE_MAX_LEN = 5


class JukeMIREmbedFunc(Protocol):
    def __call__(self, file_path: str, song_id: str) -> list[EmbeddingJukeMIR]: ...


class AuditusEmbedFunc(Protocol):
    def __call__(self, file_path: str, song_id: str) -> list[EmbeddingAuditus]: ...


class SongQueue:
    def __init__(self, name: str, q_type: QueueObject):
        self.name = name
        self.q_type = q_type
        self._process: mp.Process
        self.queue = mp.Manager().list()
        self.lock = mp.Lock()
        self.condition = mp.Condition(self.lock)

        # Debug tracking
        self._lock_holder = None
        self._lock_acquired_at = None

    def put(self, item):
        LOGGER.debug(f"[{self.name}] PID:{os.getpid()} Attempting to acquire lock for put({item})")
        with self.lock:
            self._lock_holder = f"put-{os.getpid()}"
            self._lock_acquired_at = time.time()
            LOGGER.debug(
                f"[{self.name}] PID:{os.getpid()} Lock acquired for put(), checking if item exists"
            )

            if item not in self.queue:
                self.queue.append(item)
                LOGGER.debug(
                    f"[{self.name}] PID:{os.getpid()} Item added to queue, notifying waiters"
                )
                self.condition.notify()
                self._lock_holder = None
                LOGGER.debug(
                    f"[{self.name}] PID:{os.getpid()} Successfully added item, releasing lock"
                )
                return True
            else:
                self._lock_holder = None
                LOGGER.debug(f"[{self.name}] PID:{os.getpid()} Item already exists, releasing lock")
                return False

    def get(self):
        LOGGER.debug(f"[{self.name}] PID:{os.getpid()} Attempting to acquire lock for get()")
        with self.condition:
            self._lock_holder = f"get-{os.getpid()}"
            self._lock_acquired_at = time.time()
            LOGGER.debug(
                f"[{self.name}] PID:{os.getpid()} Lock acquired for get(), queue length: {len(self.queue)}"
            )

            while len(self.queue) == 0:
                LOGGER.debug(f"[{self.name}] PID:{os.getpid()} Queue empty, waiting for items...")
                wait_start = time.time()
                if not self.condition.wait(timeout=30):  # 30 second timeout
                    wait_duration = time.time() - wait_start
                    LOGGER.warning(
                        f"[{self.name}] PID:{os.getpid()} Timeout after {wait_duration:.2f}s waiting for queue items"
                    )
                    self._lock_holder = None
                    raise TimeoutError(f"Queue {self.name} timeout after 30s")
                else:
                    wait_duration = time.time() - wait_start
                    LOGGER.debug(
                        f"[{self.name}] PID:{os.getpid()} Wait completed after {wait_duration:.2f}s, queue length now: {len(self.queue)}"
                    )

            result = self.queue.pop(0)
            self._lock_holder = None
            LOGGER.debug(
                f"[{self.name}] PID:{os.getpid()} Successfully got item: {result}, releasing lock"
            )
            return result

    def remove(self, item):
        LOGGER.debug(
            f"[{self.name}] PID:{os.getpid()} Attempting to acquire lock for remove({item})"
        )
        with self.lock:
            self._lock_holder = f"remove-{os.getpid()}"
            self._lock_acquired_at = time.time()
            LOGGER.debug(
                f"[{self.name}] PID:{os.getpid()} Lock acquired for remove(), checking if item exists"
            )

            if item in self.queue:
                self.queue.remove(item)
                LOGGER.debug(
                    f"[{self.name}] PID:{os.getpid()} Item removed from queue, releasing lock"
                )
            else:
                LOGGER.debug(
                    f"[{self.name}] PID:{os.getpid()} Item not found in queue, releasing lock"
                )

            self._lock_holder = None

    def __contains__(self, item):
        LOGGER.debug(
            f"[{self.name}] PID:{os.getpid()} Attempting to acquire lock for __contains__({item})"
        )
        with self.lock:
            self._lock_holder = f"contains-{os.getpid()}"
            self._lock_acquired_at = time.time()
            LOGGER.debug(f"[{self.name}] PID:{os.getpid()} Lock acquired for __contains__()")

            result = item in self.queue
            self._lock_holder = None
            LOGGER.debug(
                f"[{self.name}] PID:{os.getpid()} Contains check result: {result}, releasing lock"
            )
            return result

    def __len__(self):
        LOGGER.debug(f"[{self.name}] PID:{os.getpid()} Attempting to acquire lock for __len__()")
        with self.lock:
            self._lock_holder = f"len-{os.getpid()}"
            self._lock_acquired_at = time.time()
            LOGGER.debug(f"[{self.name}] PID:{os.getpid()} Lock acquired for __len__()")

            result = len(self.queue)
            self._lock_holder = None
            LOGGER.debug(f"[{self.name}] PID:{os.getpid()} Length result: {result}, releasing lock")
            return result

    def peek(self):
        LOGGER.debug(
            f"[{self.name}] PID:{os.getpid()} Starting peek() - will wait for items if needed"
        )
        while True:
            LOGGER.debug(f"[{self.name}] PID:{os.getpid()} Attempting to acquire lock for peek()")
            with self.condition:
                self._lock_holder = f"peek-{os.getpid()}"
                self._lock_acquired_at = time.time()
                LOGGER.debug(
                    f"[{self.name}] PID:{os.getpid()} Lock acquired for peek(), queue length: {len(self.queue)}"
                )

                if len(self.queue) > 0:
                    result = self.queue[0]
                    self._lock_holder = None
                    LOGGER.debug(
                        f"[{self.name}] PID:{os.getpid()} Peek successful: {result}, releasing lock"
                    )
                    return result
                else:
                    self._lock_holder = None
                    LOGGER.debug(
                        f"[{self.name}] PID:{os.getpid()} Queue empty for peek(), releasing lock and sleeping"
                    )

            LOGGER.debug(f"[{self.name}] PID:{os.getpid()} Sleeping 1s before retry peek()")
            time.sleep(1)  # Sleep outside the lock

    def peek_all(self):
        LOGGER.debug(f"[{self.name}] PID:{os.getpid()} Attempting to acquire lock for peek_all()")
        with self.lock:
            self._lock_holder = f"peek_all-{os.getpid()}"
            self._lock_acquired_at = time.time()
            LOGGER.debug(
                f"[{self.name}] PID:{os.getpid()} Lock acquired for peek_all(), queue length: {len(self.queue)}"
            )

            result = list(self.queue)
            self._lock_holder = None
            LOGGER.debug(
                f"[{self.name}] PID:{os.getpid()} Peek_all successful, returned {len(result)} items, releasing lock"
            )
            return result

    def debug_status(self):
        """Get current debug status without acquiring locks (best effort)"""
        try:
            return {
                "name": self.name,
                "lock_holder": self._lock_holder,
                "lock_duration": time.time() - self._lock_acquired_at
                if self._lock_acquired_at
                else None,
                "queue_length": len(self.queue) if hasattr(self, "queue") else "unknown",
            }
        except:
            return {"name": self.name, "status": "unable_to_get_status"}
