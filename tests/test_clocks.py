# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""StreamClocks. Run with pytest, or directly: python tests/test_clocks.py"""

import asyncio

from avaturn_live_streamer.clocks import StreamClocks


def test_wakeup_at_does_not_wake_early_after_a_shift_during_sleep() -> None:
    async def scenario() -> None:
        clocks = StreamClocks()
        clocks.start()

        async def late_frame() -> None:
            await asyncio.sleep(0.05)
            # A frame 30 ms late: the media clock moves back by 30 ms.
            async with clocks.measure_delay_after_deadline(clocks.now - 0.03):
                pass

        task = asyncio.get_running_loop().create_task(late_frame())
        await clocks.wakeup_at(0.1)
        await task
        assert clocks.now >= 0.1, f"woke at media time {clocks.now:.4f}, before 0.1"

    asyncio.run(scenario())


if __name__ == "__main__":
    test_wakeup_at_does_not_wake_early_after_a_shift_during_sleep()
    print("ok")
