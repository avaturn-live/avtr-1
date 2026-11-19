# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Worklet bridging the event bus to a local WebRTC peer (aiortc-backed).

Parallel to `worklets/rtc.py`'s `RTCWorklet` but for `LocalRTC`. Drops the
outbound SDK-message bridge (out of scope for local dev) and pulls inbound
audio push-style from `LocalRTC.recv_audio_chunk` instead of polling Daily's
speaker device.

Video frames carry their audio (`Frame.audio`), so both go out via the same
worker.
"""

import math
from asyncio import TaskGroup
from fractions import Fraction

import av
import numpy as np
from attrs import define

from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer import constant
from avaturn_live_streamer.constant import FRAME_DURATION, VIDEO_FPS, VIDEO_RESOLUTION
from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    ParticipantJoined,
    ParticipantLeft,
    Shutdown,
    StreamEnded,
    StreamStarted,
    UserSpeechReceived,
    UserSpeechStreamEnd,
    UserSpeechStreamStart,
    VideoFrameGenerated,
)
from avaturn_live_streamer.localrtc.peer import LocalRTC
from avaturn_live_streamer.types import PixelFormat
from avaturn_live_streamer.utils.async_utils import cancel_and_wait_completion
from avaturn_live_streamer.utils.exceptions import async_log_entry_exit

_LOGGER = get_logger()

_PTS_PER_FRAME_NOMINAL = 90_000 // VIDEO_FPS
_VIDEO_TIME_BASE = Fraction(1, 90_000)


def _build_video_frame(buffer: bytes, pixel_format: PixelFormat) -> av.VideoFrame:
    height, width = VIDEO_RESOLUTION
    if pixel_format.has_alpha:
        height *= 2
    arr = np.frombuffer(buffer, dtype=np.uint8).reshape(height * 3 // 2, width)
    return av.VideoFrame.from_ndarray(arr, format="yuv420p")


@define
class LocalRTCWorklet:
    _peer: LocalRTC
    _pixel_format: PixelFormat
    # False: never publish user speech from the peer, e.g. when a scripted user provides it.
    _read_user_audio: bool = True

    @async_log_entry_exit
    async def run(self, bus: EventBus, clocks: StreamClocks) -> None:
        # Block on ICE/DTLS reaching "connected" before spawning the workers.
        # Without this, sub-workers race the connection and clocks.start()
        # could anchor the wall-clock PTS epoch on the renderer's first frame
        # -- potentially many seconds before media can actually flow.
        await self._peer.wait_connected(timeout=30.0)

        async with TaskGroup() as tg:
            main_task = tg.create_task(self._video_to_peer_worker(bus.clone(), clocks))
            lifecycle_task = tg.create_task(self._participant_lifecycle_worker(bus.clone()))
            user_audio_task = (
                tg.create_task(self._read_user_audio_loop(bus, clocks))
                if self._read_user_audio
                else None
            )
            bus.ready()

            await bus.publish(ParticipantJoined(participant_id="local"))

            await main_task
            if user_audio_task is not None:
                await cancel_and_wait_completion(user_audio_task)
            # The lifecycle worker only exits on a peer state change; on a
            # bus-driven shutdown (timeout, etc.) it would block the TaskGroup
            # forever, preventing peer.close() upstream — which in turn keeps
            # the browser's mic stream feeding _consume_inbound_audio. Cancel
            # it explicitly unless it already returned naturally.
            if not lifecycle_task.done():
                await cancel_and_wait_completion(lifecycle_task)

    @async_log_entry_exit
    async def _video_to_peer_worker(self, bus: EventBus, clocks: StreamClocks) -> None:
        n_frame = 0
        epoch: float | None = None
        last_pts = 0

        async with bus.subscribe(VideoFrameGenerated, Shutdown) as sub:
            bus.ready()

            first_event = await sub.get_next()
            if isinstance(first_event, Shutdown) or first_event is None:
                return

            frame = first_event.frame
            clocks.start()
            await bus.publish(StreamStarted())

            while frame is not None:
                if epoch is None:
                    epoch = frame.timestamp

                # Hybrid PTS: take whichever is larger of nominal advance vs.
                # wall-clock since epoch. Renderer ahead of schedule -> nominal
                # (smooth 25 fps cadence on wire). Renderer stalls -> wall
                # (receiver pauses and resumes at the right real-world moment).
                nominal_next = last_pts + _PTS_PER_FRAME_NOMINAL
                wall_pts = round((frame.timestamp - epoch) * 90_000)
                pts = max(nominal_next, wall_pts)
                last_pts = pts

                vframe = _build_video_frame(frame.buffer, frame.pixel_format)
                vframe.pts = pts
                vframe.time_base = _VIDEO_TIME_BASE

                await self._peer.push_video_frame(vframe)
                if not frame.audio.is_empty:
                    await self._peer.push_audio_chunk(frame.audio)

                n_frame += 1

                # Algorithm 1, Send: the next frame is late only when now > its timestamp.
                next_frame_work_should_be_ready_at = frame.timestamp + FRAME_DURATION

                async with clocks.measure_delay_after_deadline(
                    next_frame_work_should_be_ready_at
                ) as maybe_delay:
                    next_event = await sub.get_next()
                    frame = (
                        next_event.frame if isinstance(next_event, VideoFrameGenerated) else None
                    )

                if maybe_delay.has_delay:
                    _LOGGER.warning(
                        "LocalRTC worklet frame delayed: delay=%.4f frame=%d",
                        maybe_delay.delay,
                        n_frame,
                        delay=maybe_delay.delay,
                        frame=n_frame,
                    )
                else:
                    await clocks.wakeup_at(next_frame_work_should_be_ready_at)

            await bus.publish(StreamEnded(duration_seconds=int(math.ceil(clocks.now))))

    async def _participant_lifecycle_worker(self, bus: EventBus) -> None:
        bus.ready()
        joined = self._peer.connection_state == "connected"
        while True:
            state = await self._peer.wait_state_change()
            if state == "connected" and not joined:
                await bus.publish(ParticipantJoined(participant_id="local"))
                joined = True
            elif state in ("closed", "failed", "disconnected") and joined:
                await bus.publish(ParticipantLeft(participant_id="local"))
                await bus.publish(Shutdown(reason="user_absent_timeout"))
                return

    async def _read_user_audio_loop(self, bus: EventBus, clocks: StreamClocks) -> None:
        """Mirror of RTCWorklet._read_user_audio_loop, push-driven by LocalRTC."""

        chunk_frame_count = constant.NATIVE_SPEECH_SAMPLE_RATE // 50  # 20ms
        chunk_duration = Fraction(chunk_frame_count, constant.NATIVE_SPEECH_SAMPLE_RATE)

        while True:
            chunk = await self._peer.recv_audio_chunk()
            if chunk.is_empty:
                continue
            next_chunk_at = clocks.now + chunk_duration * 1.5

            _LOGGER.info("User audio stream started (data available)")
            await bus.publish(UserSpeechStreamStart())

            while True:
                await bus.publish(UserSpeechReceived(buffer=chunk))
                try:
                    async with clocks.wait_until(next_chunk_at):
                        chunk = await self._peer.recv_audio_chunk()
                except TimeoutError:
                    break
                next_chunk_at += chunk_duration
            _LOGGER.info("User audio stream ended (no more data)")
            await bus.publish(UserSpeechStreamEnd())
