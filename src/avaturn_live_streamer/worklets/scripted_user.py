# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Scripted user for measurement runs with real voice agents.

The worklet takes the place of the browser microphone. It publishes the same bus events as the
local WebRTC peer: UserSpeechStreamStart, UserSpeechReceived (20 ms chunks at real-time pace) and
UserSpeechStreamEnd. The audio stream is continuous, with silence between utterances, because the
voice agents detect the end of a turn from silence in the stream.

The run:
    1. Say the prompt once ("Please count from one to ten.").
    2. Per turn: wait until the reply starts to play (SegmentPlaybackStarted), then, after a random
       delay, say the interruption ("Stop. Count again."). The agent stops (the
       ILC), and its answer to the interruption is the next turn's reply (the next RLC). So no
       turn waits for an extra answer or an idle gap. The avatar is silent when that answer
       arrives, as the RLC definition requires.
    3. After the last turn, wait until playback is idle.
Every played segment is recorded, with the RLC and, for an interrupted segment, the ILC. Wall-clock totals are measured as in the scripted agent: from
the arrival in the scheduler to the playback event, stamped on receipt.
"""

import asyncio
import contextlib
import random
from fractions import Fraction
from typing import Any

import attrs

from avaturn_live_streamer import constant
from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    SegmentPlaybackCancelled,
    SegmentPlaybackCompleted,
    SegmentPlaybackInterrupted,
    SegmentPlaybackStarted,
    Shutdown,
    TurnLatencies,
    UserSpeechReceived,
    UserSpeechStreamEnd,
    UserSpeechStreamStart,
)
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.utils.exceptions import async_log_entry_exit

_LOGGER = get_logger()

_CHUNK = constant.NATIVE_SPEECH_SAMPLE_RATE // 50  # 20 ms, as the local WebRTC peer
_CHUNK_SECONDS = Fraction(_CHUNK, constant.NATIVE_SPEECH_SAMPLE_RATE)  # exact, as in the scheduler


def _measured(latencies: TurnLatencies | None, end_wall: float | None) -> dict[str, Any] | None:
    if latencies is None:
        return None
    return {
        "t_wait": latencies.t_wait,
        "t_pad": latencies.t_pad,
        "t_lead": latencies.t_lead,
        "t_present": latencies.t_present,
        "t_total": latencies.t_total,
        "t_total_wall": None if end_wall is None else end_wall - latencies.arrival_monotonic,
        "no_fragment_at_admit": latencies.no_fragment_at_admit,
        "consistent": latencies.consistent,
    }


@attrs.define
class _Segment:
    turn: int
    role: str  # "reply": the first segment after the prompt; "other": any later one
    started: tuple[TurnLatencies | None, float] | None = None
    completed_wall: float | None = None
    interrupted: TurnLatencies | None = None


@attrs.define
class ScriptedUser:
    """Worklet that plays a fixed conversation against a real voice agent, then shuts down."""

    _prompt: SpeechBuffer
    _interruption: SpeechBuffer
    _turns: int
    _interrupt_delay: tuple[float, float] = (0.6, 1.8)  # seconds after the reply starts to play
    _idle_gap: float = 2.0  # seconds without playback before the next turn
    _reply_timeout: float = 20.0  # seconds to wait for a reply
    # Seconds to wait for the agent's answer to the interruption. The agents need 3-4 s from the
    # end of the user's speech to their first audio, so a fixed short pause is not enough.
    _answer_timeout: float = 8.0
    _seed: int = 0
    # One row per played segment. The caller reads it after run.
    results: list[dict[str, Any]] = attrs.Factory(list)

    _queue: asyncio.Queue = attrs.field(init=False, factory=asyncio.Queue)
    _segments: dict[str, _Segment] = attrs.field(init=False, factory=dict)
    _turn: int = attrs.field(init=False, default=-1)
    _reply_started: asyncio.Event = attrs.field(init=False, factory=asyncio.Event)
    # True from the end of the prompt until the reply starts: only then is a new segment the reply.
    _awaiting_reply: bool = attrs.field(init=False, default=False)
    _segment_started: asyncio.Event = attrs.field(init=False, factory=asyncio.Event)
    _playing: set[str] = attrs.field(init=False, factory=set)
    _last_playback_change: float = attrs.field(init=False, default=0.0)

    def __attrs_post_init__(self) -> None:
        rate = constant.NATIVE_SPEECH_SAMPLE_RATE
        self._prompt = self._prompt.resample(rate)
        self._interruption = self._interruption.resample(rate)

    @async_log_entry_exit
    async def run(self, bus: EventBus, clocks: StreamClocks) -> None:
        async with asyncio.TaskGroup() as tg:
            listener = tg.create_task(self._listen(bus.clone(), clocks))
            microphone = tg.create_task(self._microphone(bus.clone(), clocks))
            try:
                await self._drive(bus, clocks)
            finally:
                listener.cancel()
                microphone.cancel()

    async def _say(self, speech: SpeechBuffer) -> None:
        """Queue an utterance and wait until the microphone has sent all of it."""
        done = asyncio.get_running_loop().create_future()
        await self._queue.put((speech, done))
        await done

    async def _microphone(self, bus: EventBus, clocks: StreamClocks) -> None:
        """Send 20 ms chunks at real-time pace: the queued utterance, else silence."""
        bus.ready()
        await clocks.wakeup_at(0)
        silence = SpeechBuffer.silence(_CHUNK_SECONDS, constant.NATIVE_SPEECH_SAMPLE_RATE)
        await bus.publish(UserSpeechStreamStart())
        current: SpeechBuffer | None = None
        done: asyncio.Future | None = None
        offset = Fraction(0)
        next_at = clocks.now
        try:
            while True:
                if current is None and not self._queue.empty():
                    current, done = self._queue.get_nowait()
                    offset = Fraction(0)
                if current is not None:
                    chunk = current.slice(offset, offset + _CHUNK_SECONDS)
                    offset += _CHUNK_SECONDS
                    if chunk.duration < _CHUNK_SECONDS:
                        rest = SpeechBuffer.silence(
                            _CHUNK_SECONDS - chunk.duration, constant.NATIVE_SPEECH_SAMPLE_RATE
                        )
                        chunk = SpeechBuffer.concat([chunk, rest])
                    if offset >= current.duration:
                        assert done is not None
                        done.set_result(None)
                        current, done = None, None
                else:
                    chunk = silence
                await bus.publish(UserSpeechReceived(buffer=chunk))
                next_at += float(_CHUNK_SECONDS)
                await clocks.wakeup_at(next_at)
        finally:
            await bus.publish(UserSpeechStreamEnd())

    async def _listen(self, bus: EventBus, clocks: StreamClocks) -> None:
        """Stamp each playback event with the wall clock on arrival."""
        async with bus.subscribe(
            SegmentPlaybackStarted,
            SegmentPlaybackCompleted,
            SegmentPlaybackInterrupted,
            SegmentPlaybackCancelled,
        ) as sub:
            bus.ready()
            async for event in sub:
                now_wall = clocks.now_wall
                loop_now = asyncio.get_running_loop().time()
                sid = str(event.segment_id)
                match event:
                    case SegmentPlaybackStarted(latencies=lat):
                        # A segment that starts before the prompt has been spoken in full answers
                        # something earlier (for example the last interruption): not the reply.
                        role = "reply" if self._awaiting_reply else "other"
                        segment = self._segments.setdefault(sid, _Segment(self._turn, role))
                        segment.started = (lat, now_wall)
                        self._playing.add(sid)
                        self._segment_started.set()
                        if self._awaiting_reply:
                            self._awaiting_reply = False
                            self._reply_started.set()
                    case SegmentPlaybackCompleted():
                        segment = self._segments.setdefault(sid, _Segment(self._turn, "other"))
                        segment.completed_wall = now_wall
                        self._playing.discard(sid)
                    case SegmentPlaybackInterrupted(latencies=lat):
                        self._segments.setdefault(
                            sid, _Segment(self._turn, "other")
                        ).interrupted = lat
                    case SegmentPlaybackCancelled():
                        self._playing.discard(sid)
                self._last_playback_change = loop_now

    async def _wait_idle(self) -> None:
        loop = asyncio.get_running_loop()
        while self._playing or loop.time() - self._last_playback_change < self._idle_gap:
            await asyncio.sleep(0.1)

    async def _drive(self, bus: EventBus, clocks: StreamClocks) -> None:
        bus.ready()
        await clocks.wakeup_at(0)
        rng = random.Random(self._seed)
        self._last_playback_change = asyncio.get_running_loop().time()
        await self._wait_idle()
        utterance = self._prompt
        for turn in range(self._turns):
            self._turn = turn
            self._reply_started.clear()
            _LOGGER.info("Scripted user: request", turn=turn)
            await self._say(utterance)
            self._awaiting_reply = True
            try:
                await asyncio.wait_for(self._reply_started.wait(), self._reply_timeout)
            except TimeoutError:
                # No reply: start this turn again with the plain prompt, from silence.
                _LOGGER.warning("Scripted user: no reply", turn=turn)
                self._awaiting_reply = False
                await self._wait_idle()
                utterance = self._prompt
                continue
            delay = rng.uniform(*self._interrupt_delay)
            await asyncio.sleep(delay)
            _LOGGER.info("Scripted user: interruption", turn=turn, delay=round(delay, 3))
            # The interruption also asks for the next reply.
            utterance = self._interruption
        # Interrupt the last reply too, then wait for its answer and for silence.
        await self._say(self._interruption)
        self._segment_started.clear()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._segment_started.wait(), self._answer_timeout)
        await self._wait_idle()
        await self._collect()
        await bus.publish(Shutdown())

    async def _collect(self) -> None:
        # The Interrupted event comes one step after playback stops: give it time to arrive.
        await asyncio.sleep(1.0)
        for sid, s in self._segments.items():
            if s.started is None:
                continue
            lat, started_wall = s.started
            self.results.append(
                {
                    "turn": s.turn,
                    "segment_id": sid,
                    "role": s.role,
                    "rlc": _measured(lat, started_wall),
                    "ilc": _measured(s.interrupted, s.completed_wall),
                    "end": "interrupted" if s.interrupted is not None else "completed",
                }
            )
