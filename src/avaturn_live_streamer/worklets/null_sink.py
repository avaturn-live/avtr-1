# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Frame sender without a transport, for offline runs such as the scripted latency sweep.

The pacing is a copy of `LocalRTCWorklet._video_to_peer_worker`: the first frame starts the clock,
each frame waits for its timestamp, and a late frame moves the clock back by its delay (delta in
Algorithm 1). Only the push to the peer is missing. Keep the two in sync.
"""

import math

from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.constant import FRAME_DURATION
from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import Shutdown, StreamEnded, StreamStarted, VideoFrameGenerated
from avaturn_live_streamer.utils.exceptions import async_log_entry_exit

_LOGGER = get_logger()


@async_log_entry_exit
async def run_null_sink_worklet(bus: EventBus, clocks: StreamClocks) -> None:
    n_frame = 0
    async with bus.subscribe(VideoFrameGenerated, Shutdown) as sub:
        bus.ready()

        first_event = await sub.get_next()
        if not isinstance(first_event, VideoFrameGenerated):
            return

        frame = first_event.frame
        clocks.start()
        await bus.publish(StreamStarted())

        while frame is not None:
            n_frame += 1
            # Algorithm 1, Send: the next frame is late only when now > its timestamp.
            next_frame_work_should_be_ready_at = frame.timestamp + FRAME_DURATION

            async with clocks.measure_delay_after_deadline(
                next_frame_work_should_be_ready_at
            ) as maybe_delay:
                next_event = await sub.get_next()
                frame = next_event.frame if isinstance(next_event, VideoFrameGenerated) else None

            if maybe_delay.has_delay:
                _LOGGER.warning(
                    "Null sink frame delayed",
                    delay=round(maybe_delay.delay, 4),
                    frame=n_frame,
                )
            else:
                await clocks.wakeup_at(next_frame_work_should_be_ready_at)

        await bus.publish(StreamEnded(duration_seconds=math.ceil(clocks.now)))
