# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Scripted voice agent for the controlled latency sweep.

The agent is a plain worklet. It has no network connection. It publishes the same bus events as
the commercial integrations: SegmentGenerationStarted, SegmentChunkGenerated,
SegmentGenerationCompleted and, for an interruption, DiscardAvatarSpeechBuffer. The speech comes
from a recording (`speech`): each turn plays its own slice of it, cut into the fragments. Without a
recording, the agent sends silence. Latency depends only on timing, so the content does not change
the results.

Phase control
    The agent uses the same StreamClocks as the rendering loop. The loop runs Step k at media time
    k * T_present - T_lead (`worklets/rendering.py`). The agent puts each event at a set offset
    before one of these instants. This sets the arrival phase T_wait of a reply, and the phase of an
    interruption.

Arrival pattern of one segment
    1. At the arrival time, the start marker and a first fragment of `first_fragment` speech.
       This is all the speech that the admitting step sees, so it alone sets T_pad.
    2. Later fragments of `fragment` speech, from AFTER_ADMIT after the admitting step on, paced
       at `rate` times real time: fragment j is sent when (speech of fragments 1..j-1) / rate has
       elapsed since the first one. `rate=None` sends all later fragments at once.
       Their size therefore cannot change T_pad.
    3. The end marker after the last fragment.
    The real-time condition of the paper needs first_fragment >= t_wait + after_admit and
    rate >= 1. The agent rejects a turn that breaks the first part. With rate >= 2, no step
    blocks. A rate below 1 delays fragments on purpose, so steps block.

Prediction (paper, Eq. for the RLC, with the conditions of the paper)
    RLC = T_lead + T_future + T_wait + T_pad, T_pad = [T_present - R]_+,
    where R is the speech in the queue when Step admits the segment.
    ILC = T_lead + T_future + T_wait of the interruption.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from typing import Any, Literal

import attrs

from avaturn_live_streamer import constant
from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    DiscardAvatarSpeechBuffer,
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    SegmentPlaybackCancelled,
    SegmentPlaybackCompleted,
    SegmentPlaybackInterrupted,
    SegmentPlaybackStarted,
    Shutdown,
    TurnLatencies,
)
from avaturn_live_streamer.management.types import SegmentId
from avaturn_live_streamer.renderer.models import RenderingModelDurations
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.utils.exceptions import async_log_entry_exit

_LOGGER = get_logger()

# Later fragments start this long after the admitting step. The step runs up to about 2 ms late
# (measured T_lead is 98-99 ms), so the margin makes sure the step does not see them.
AFTER_ADMIT = Fraction(10, 1000)


@dataclass(frozen=True, slots=True)
class ScriptedTurn:
    """One reply of the scripted agent. All durations are in seconds."""

    # Time from the arrival of the start marker and first fragment to the admitting step.
    # Must be in (0, T_present). Values near 0 race with the step: keep at least a few ms.
    t_wait: Fraction
    # Speech in the first fragment, sent together with the start marker.
    first_fragment: Fraction
    # Speech in each later fragment. The last fragment can be shorter.
    fragment: Fraction
    # Total speech in the segment.
    duration: Fraction = Fraction(3)
    # Pace of the later fragments as a multiple of real time. None sends them all at arrival.
    rate: Fraction | None = Fraction(2)
    # Number of steps after the admitting step at which the interruption arrives.
    # None lets the segment play to its end.
    interrupt_after_steps: int | None = 4
    # Time from the interruption to the step that admits it. Must be in (0, T_present).
    interrupt_t_wait: Fraction = Fraction(1, 10)
    # Free text copied to the result row, for example the name of a sweep set.
    label: str = ""

    def __post_init__(self) -> None:
        if self.first_fragment <= 0 or self.fragment <= 0:
            raise ValueError("Fragment durations must be positive")
        if self.first_fragment > self.duration:
            raise ValueError("first_fragment must not be longer than duration")
        if self.rate is not None and self.rate <= 0:
            raise ValueError("rate must be positive or None")
        if self.interrupt_after_steps is not None and self.interrupt_after_steps < 1:
            raise ValueError("interrupt_after_steps must be at least 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "t_wait": float(self.t_wait),
            "first_fragment": float(self.first_fragment),
            "fragment": float(self.fragment),
            "duration": float(self.duration),
            "rate": None if self.rate is None else float(self.rate),
            "interrupt_after_steps": self.interrupt_after_steps,
            "interrupt_t_wait": float(self.interrupt_t_wait),
            "label": self.label,
        }


_ActionKind = Literal["fragment", "complete", "interrupt"]


@dataclass(frozen=True, slots=True)
class _Action:
    at: Fraction
    kind: _ActionKind
    speech: Fraction = Fraction(0)


def _fragment_schedule(turn: ScriptedTurn) -> list[tuple[Fraction, Fraction]]:
    """Return (send offset from the first later fragment, speech) for each later fragment."""
    schedule = list[tuple[Fraction, Fraction]]()
    sent_later = Fraction(0)
    while turn.first_fragment + sent_later < turn.duration:
        speech = min(turn.fragment, turn.duration - turn.first_fragment - sent_later)
        offset = Fraction(0) if turn.rate is None else sent_later / turn.rate
        schedule.append((offset, speech))
        sent_later += speech
    return schedule


def _audio_sample_aligned(duration: Fraction) -> Fraction:
    """Round a duration down to whole samples at the native rate."""
    rate = constant.NATIVE_SPEECH_SAMPLE_RATE
    return Fraction(int(duration * rate), rate)


@attrs.define
class ScriptedAgent:
    """Worklet that plays a list of ScriptedTurn objects, one after the other, then shuts down."""

    _turns: Sequence[ScriptedTurn]
    _durations: RenderingModelDurations
    # Idle steps between the end of one turn's playback and the next arrival. At least 2, so the
    # buffer holds exactly T_future of silence when the next segment arrives.
    _gap_steps: int = 3
    _segment_prefix: str = "scripted"
    # Recorded speech, resampled to the native rate on load. None sends silence.
    _speech: SpeechBuffer | None = attrs.field(default=None, alias="speech")
    # Later fragments start this long after the admitting step (see AFTER_ADMIT).
    _after_admit: Fraction = attrs.field(default=AFTER_ADMIT, alias="after_admit")
    # One row per turn: configuration, prediction and measurement. The caller reads it after run.
    results: list[dict[str, Any]] = attrs.Factory(list)

    def __attrs_post_init__(self) -> None:
        if self._speech is not None:
            self._speech = self._speech.resample(constant.NATIVE_SPEECH_SAMPLE_RATE)
            longest = max(turn.duration for turn in self._turns)
            if self._speech.duration < longest:
                raise ValueError(f"speech is shorter than the longest turn ({float(longest)} s)")
        if self._gap_steps < 2:
            raise ValueError("gap_steps must be at least 2")
        tp = self._durations.present
        for turn in self._turns:
            if not 0 < turn.t_wait < tp or not 0 < turn.interrupt_t_wait < tp:
                raise ValueError(f"t_wait and interrupt_t_wait must be in (0, {float(tp)}) s")
            if turn.first_fragment < turn.t_wait + self._after_admit:
                raise ValueError(
                    f"first_fragment {float(turn.first_fragment)} s breaks the real-time condition:"
                    f" use at least t_wait + {float(self._after_admit)} s"
                )
            # The interruption must arrive while the segment still plays. Playback starts at most
            # T_future + T_present after the admitted step's present window starts.
            m = turn.interrupt_after_steps
            if m is not None and turn.duration < (m + 3) * tp:
                raise ValueError(
                    f"duration {float(turn.duration)} s is too short for an interruption "
                    f"{m} steps after admission: use at least {float((m + 3) * tp)} s"
                )

    def _step_time(self, k: int) -> Fraction:
        """Media time at which the rendering loop runs Step k."""
        return k * self._durations.present - self._durations.lead

    def _first_step_after(self, t: Fraction) -> int:
        return ceil((t + self._durations.lead) / self._durations.present)

    def _plan(
        self, turn: ScriptedTurn, admit_step: int
    ) -> tuple[Fraction, list[_Action], Fraction]:
        """Return the arrival time, the timed actions and the speech in the queue at admission."""
        admit_at = self._step_time(admit_step)
        arrival = admit_at - turn.t_wait

        actions = list[_Action]()
        last_at = arrival
        for offset, speech in _fragment_schedule(turn):
            last_at = admit_at + self._after_admit + offset
            actions.append(_Action(last_at, "fragment", speech))
        actions.append(_Action(last_at, "complete"))
        received_at_admit = turn.first_fragment

        if turn.interrupt_after_steps is not None:
            interrupt_at = self._step_time(admit_step + turn.interrupt_after_steps)
            actions.append(_Action(interrupt_at - turn.interrupt_t_wait, "interrupt"))

        # Stable sort: at equal times, fragments and completion go before the interruption.
        actions.sort(key=lambda a: a.at)
        return arrival, actions, received_at_admit

    def _predict(self, turn: ScriptedTurn, received_at_admit: Fraction) -> dict[str, float]:
        d = self._durations
        t_pad = max(Fraction(0), d.present - received_at_admit)
        rlc = d.lead + d.future + turn.t_wait + t_pad
        prediction = {
            "received_at_admit": float(received_at_admit),
            "rlc_t_pad": float(t_pad),
            "rlc": float(rlc),
        }
        if turn.interrupt_after_steps is not None:
            prediction["ilc"] = float(d.lead + d.future + turn.interrupt_t_wait)
        return prediction

    @staticmethod
    def _measured(latencies: TurnLatencies | None, end_wall: float | None) -> dict[str, Any] | None:
        if latencies is None:
            return None
        return {
            "t_wait": latencies.t_wait,
            "t_pad": latencies.t_pad,
            "t_lead": latencies.t_lead,
            "t_present": latencies.t_present,
            "t_total": latencies.t_total,
            # Wall clock from the arrival to the playback event that ends the latency.
            "t_total_wall": None if end_wall is None else end_wall - latencies.arrival_monotonic,
            "no_fragment_at_admit": latencies.no_fragment_at_admit,
            "consistent": latencies.consistent,
        }

    @async_log_entry_exit
    async def run(self, bus: EventBus, clocks: StreamClocks) -> None:
        playback: dict[SegmentId, _Playback] = {}
        async with asyncio.TaskGroup() as tg:
            listener = tg.create_task(self._listen_playback(bus.clone(), clocks, playback))
            try:
                await self._drive(bus, clocks, playback)
            finally:
                listener.cancel()

    async def _listen_playback(
        self, bus: EventBus, clocks: StreamClocks, playback: dict[SegmentId, "_Playback"]
    ) -> None:
        """Stamp each playback event with the wall clock when it arrives.

        The driver sleeps until its next timed action, so it cannot read events on time itself.
        """
        async with bus.subscribe(
            SegmentPlaybackStarted,
            SegmentPlaybackCompleted,
            SegmentPlaybackInterrupted,
            SegmentPlaybackCancelled,
        ) as sub:
            bus.ready()
            async for event in sub:
                now_wall = clocks.now_wall
                state = playback.setdefault(event.segment_id, _Playback())
                match event:
                    case SegmentPlaybackStarted(latencies=lat):
                        state.started = (lat, now_wall)
                    case SegmentPlaybackCompleted():
                        # For an interrupted segment, this is the moment playback stops.
                        state.completed_wall = now_wall
                        if not state.expect_interrupt:
                            state.done.set()
                    case SegmentPlaybackInterrupted(latencies=lat):
                        # The scheduler emits this one step after playback stops.
                        state.interrupted = lat
                        state.done.set()
                    case SegmentPlaybackCancelled():
                        state.cancelled = True
                        state.done.set()

    async def _drive(
        self, bus: EventBus, clocks: StreamClocks, playback: dict[SegmentId, "_Playback"]
    ) -> None:
        bus.ready()
        # Wait until the frame sender starts the clock. Before that, `clocks.now` is 0.
        await clocks.wakeup_at(0)

        for index, turn in enumerate(self._turns):
            segment_id = SegmentId(f"{self._segment_prefix}-{index}")
            earliest = Fraction(clocks.now) + self._gap_steps * self._durations.present
            admit_step = self._first_step_after(earliest + turn.t_wait)
            arrival, actions, received_at_admit = self._plan(turn, admit_step)
            interrupted = turn.interrupt_after_steps is not None
            state = playback.setdefault(segment_id, _Playback())
            state.expect_interrupt = interrupted

            row: dict[str, Any] = {
                "index": index,
                "segment_id": segment_id,
                "config": turn.as_dict(),
                "admit_step": admit_step,
                "predicted": self._predict(turn, received_at_admit),
            }
            _LOGGER.info("Scripted turn planned", segment_id=segment_id, **row["config"])

            audio = self._turn_audio(index, turn)
            cursor = Fraction(0)

            await clocks.wakeup_at(float(arrival))
            await bus.publish(SegmentGenerationStarted(segment_id=segment_id))
            await self._publish_speech(bus, segment_id, audio, cursor, turn.first_fragment)
            cursor += turn.first_fragment

            for action in actions:
                await clocks.wakeup_at(float(action.at))
                if action.kind == "fragment":
                    await self._publish_speech(bus, segment_id, audio, cursor, action.speech)
                    cursor += action.speech
                elif action.kind == "complete":
                    await bus.publish(SegmentGenerationCompleted(segment_id=segment_id))
                else:
                    # Same order as the Cartesia client: Completed before Discard, so the
                    # scheduler releases the write-active segment. A completed segment ignores
                    # the second Completed with a warning, so send it only when needed.
                    if not any(a.kind == "complete" and a.at <= action.at for a in actions):
                        await bus.publish(SegmentGenerationCompleted(segment_id=segment_id))
                    await bus.publish(DiscardAvatarSpeechBuffer())
                    break  # The agent stops generating on an interruption.

            await state.done.wait()
            del playback[segment_id]

            started_lat, started_wall = state.started or (None, None)
            row["rlc"] = self._measured(started_lat, started_wall)
            row["ilc"] = self._measured(state.interrupted, state.completed_wall)
            row["end"] = (
                "cancelled" if state.cancelled else "interrupted" if interrupted else "completed"
            )
            self.results.append(row)
            _LOGGER.info(
                "Scripted turn done",
                segment_id=segment_id,
                predicted_rlc=row["predicted"]["rlc"],
                measured_rlc_wall=row["rlc"] and row["rlc"]["t_total_wall"],
                predicted_ilc=row["predicted"].get("ilc"),
                measured_ilc_wall=row["ilc"] and row["ilc"]["t_total_wall"],
            )

        await bus.publish(Shutdown())

    def _turn_audio(self, index: int, turn: ScriptedTurn) -> SpeechBuffer:
        """Return the speech of one turn: a slice of the recording, or silence."""
        duration = _audio_sample_aligned(turn.duration)
        if self._speech is None:
            return SpeechBuffer.silence(duration, constant.NATIVE_SPEECH_SAMPLE_RATE)
        # Consecutive turns take consecutive slices and wrap at the end of the recording.
        span = self._speech.duration - duration
        start = _audio_sample_aligned((index * duration) % span) if span > 0 else Fraction(0)
        return self._speech.slice(start, start + duration)

    async def _publish_speech(
        self,
        bus: EventBus,
        segment_id: SegmentId,
        audio: SpeechBuffer,
        start: Fraction,
        speech: Fraction,
    ) -> None:
        start = _audio_sample_aligned(start)
        buffer = audio.slice(start, start + _audio_sample_aligned(speech))
        await bus.publish(SegmentChunkGenerated(segment_id=segment_id, buffer=buffer))


@attrs.define
class _Playback:
    """Playback events of one segment, stamped with the wall clock on arrival."""

    expect_interrupt: bool = False
    started: tuple[TurnLatencies | None, float] | None = None
    completed_wall: float | None = None
    interrupted: TurnLatencies | None = None
    cancelled: bool = False
    done: asyncio.Event = attrs.Factory(asyncio.Event)
