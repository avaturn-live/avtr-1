# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import attrs

from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.utils.async_utils import aiotime

_LOGGER = get_logger()


@attrs.define
class _MaybeDelay:
    _delay: float | None = None

    @property
    def has_delay(self) -> bool:
        assert self._delay is not None
        return self._delay > 0.0

    @property
    def delay(self) -> float:
        assert self._delay is not None
        return self._delay


class StreamClocks:
    def __init__(self):
        self._total_delay = 0.0
        self._total_sent_duration = 0.0
        self._started_at: float | None = None
        self._started = asyncio.Event()

    def start(self):
        self._started_at = aiotime()
        self._started.set()

    async def wakeup_at(self, time: float):
        """Sleep until `now >= time` on the media clock.

        A late frame can shift the clock back (delta in Algorithm 1) while this coroutine sleeps.
        A sleep computed once then ends early on the media clock. So sleep again for the rest,
        until the media clock reaches `time`.
        """
        await self._started.wait()
        cn = self.now
        remaining = time - cn
        if remaining <= 0:
            if remaining < 0:
                _LOGGER.warning(
                    "StreamClock: negative sleep sleep_for=%.4f delays=%.4f compensated_now=%.4f",
                    remaining,
                    self._total_delay,
                    cn,
                )
            # Already due: still yield once to the event loop, as before.
            await asyncio.sleep(0)
            return

        while remaining > 0:
            await asyncio.sleep(remaining)
            remaining = time - self.now

    @property
    def now(self):
        return (
            aiotime() - self._total_delay - self._started_at
            if self._started_at is not None
            else 0.0
        )

    @property
    def now_wall(self) -> float:
        """Raw monotonic time. Unlike `now`, it is not shifted by the delays the clock absorbed."""
        return aiotime()

    @property
    def started(self) -> bool:
        return self._started_at is not None

    @asynccontextmanager
    async def measure_delay_after_deadline(
        self, deadline_at: float
    ) -> AsyncGenerator[_MaybeDelay, None]:
        maybe_delay = _MaybeDelay()
        try:
            yield maybe_delay
        finally:
            delay = max(0, self.now - deadline_at)
            maybe_delay._delay = delay
            self._total_delay += delay

    @asynccontextmanager
    async def wait_until(self, deadline_at: float) -> AsyncGenerator[None, None]:
        await self._started.wait()
        async with asyncio.timeout(deadline_at - self.now):
            yield
