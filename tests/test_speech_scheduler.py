# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""SpeechScheduler against Algorithm 2 of the paper.

Run with pytest, or directly: python tests/test_speech_scheduler.py
"""

import asyncio
from fractions import Fraction

import numpy as np

from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.speech.speech_scheduler import SegmentStarted, SpeechScheduler

SR = 24_000
TP = Fraction(200, 1000)
TF = Fraction(205, 1000)


def _ms(v: int) -> Fraction:
    return Fraction(v, 1000)


def _samples(buf: SpeechBuffer) -> np.ndarray:
    return np.frombuffer(buf.to_bytes(), dtype=np.int16)


def _speech(duration: Fraction, value: int) -> SpeechBuffer:
    return SpeechBuffer.full(value, duration, SR)


def _scheduler() -> SpeechScheduler:
    return SpeechScheduler(
        sample_rate=SR, present_duration=TP, future_duration=TF, clocks=StreamClocks()
    )


def test_segment_that_starts_and_ends_in_one_step_is_right_padded() -> None:
    """Algorithm 2, line 15 left-pads only an active segment. A segment that ends in its start
    step is complete, so line 16 pads it on the right and adds no T_pad."""

    async def scenario() -> None:
        s = _scheduler()
        await s.do_step()  # idle step: B holds T_future of silence

        await s.start_segment("short")
        await s.append(_speech(_ms(100), 1000), segment_id="short")
        await s.end_segment("short")
        step = await s.do_step()

        # B = [205 committed silence][100 speech][100 right pad]. The speech follows the committed
        # window directly.
        window = np.concatenate([_samples(step.present), _samples(step.future)])
        speech = np.flatnonzero(window == 1000)
        assert len(speech) == 100 * SR // 1000
        assert speech[0] == TF * SR, "speech must follow the committed window (no left pad)"
        assert speech[-1] == len(window) - 1 - 100 * SR // 1000, "silence must end the window"

        # No pad moves the start event: present start + T_future.
        started = [e for e in s._event_queue if isinstance(e.event, SegmentStarted)]
        started += [e for e in step.events if isinstance(e.event, SegmentStarted)]
        assert [e.timestamp for e in started] == [TP + TF]

    asyncio.run(scenario())


def test_interrupt_during_r3_wait_discards_partly_filled_window() -> None:
    """K2: Interrupt clears A (Algorithm 2, line 21) while Step waits on R3 with A partly filled."""

    async def scenario() -> None:
        s = _scheduler()
        await s.do_step()  # idle step

        await s.start_segment("reply")
        await s.append(_speech(_ms(500), 1000), segment_id="reply")
        await s.do_step()  # admits the segment, 500 ms overshoots the window
        await s.do_step()  # uses the overage, no wait

        blocked = asyncio.create_task(s.do_step())  # needs more audio: waits on R3
        await asyncio.sleep(0)
        assert not blocked.done(), "step must wait on R3"

        # A partial fill: it goes into A, and the step keeps waiting for the rest.
        await s.append(_speech(_ms(50), 7777), segment_id="reply")
        await asyncio.sleep(0)
        assert not blocked.done()

        await s.interrupt()
        step = await asyncio.wait_for(blocked, timeout=1)

        played = [_samples(step.present), _samples(step.future)]
        for _ in range(4):
            nxt = await s.do_step()
            played += [_samples(nxt.present)]
        assert not any((p == 7777).any() for p in played), "discarded audio must not play"

        # After the interruption, the window holds the kept T_future of speech, then silence.
        window = np.concatenate([_samples(step.present), _samples(step.future)])
        assert len(window) == (TP + TF) * SR

    asyncio.run(scenario())


if __name__ == "__main__":
    test_segment_that_starts_and_ends_in_one_step_is_right_padded()
    test_interrupt_during_r3_wait_discards_partly_filled_window()
    print("ok")
