#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Measure the RLC and ILC with a real voice agent and a scripted user (no person needed).

The scripted user (worklets/scripted_user.py) asks the agent to count from one to ten, then
interrupts it after a random delay with "Stop. Count again." Each turn gives one
RLC (the reply) and one ILC (its interruption). The agent's answer to the interruption is the
next turn's reply.

Keys come from env files (KEY=VALUE) and are never printed: OPENAI_API_KEY or OPENAI__API_KEY,
CARTESIA_API_KEY and CARTESIA_AGENT_ID. The OpenAI session gets AGENT_PROMPT. A Cartesia agent
has its prompt in its deployment: give it the same text.

Usage:
    python scripts/make_user_prompts.py --env-file ../avaturn-live-core-backend/local.env
    python scripts/run_commercial_agents.py --engine openai --turns 50 \\
        --env-file ../avaturn-live-core-backend/local.env --out runs/openai.jsonl
    python scripts/run_commercial_agents.py --engine cartesia --turns 50 \\
        --env-file ../avaturn-live-core-backend/override.env --out runs/cartesia.jsonl
    # With the real renderer: add --renderer-url http://localhost:8000 --sink webrtc
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean

from aiortc import RTCPeerConnection
from make_user_prompts import openai_key, read_env_files
from run_scripted_sweep import (
    TIMINGS,
    _durations,
    _pick_avatar,
    _start_receiver,
)

from avaturn_live_streamer.config import RendererConfig
from avaturn_live_streamer.conversation_engines.builders import (
    CartesiaEngineOptions,
    OpenAIEngineOptions,
    build_engine,
)
from avaturn_live_streamer.core.logging_config import LoggingConfig
from avaturn_live_streamer.core.logs import setup_logging
from avaturn_live_streamer.localrtc import LocalRTC, LocalRTCWorklet
from avaturn_live_streamer.renderer.client import RendererClient
from avaturn_live_streamer.renderer.interface import AbstractRendererClient
from avaturn_live_streamer.renderer.mock import MockRendererClient, _make_frames, fixed_frame_times
from avaturn_live_streamer.renderer.registry import RendererClientRegistry
from avaturn_live_streamer.runner import run_stream
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.types import BackgroundId, PixelFormat, RendererAvatarId
from avaturn_live_streamer.worklets.delayed_event import run_delayed_event_worklet
from avaturn_live_streamer.worklets.latency import LatencyWorklet
from avaturn_live_streamer.worklets.null_sink import run_null_sink_worklet
from avaturn_live_streamer.worklets.rendering import RenderingWorklet, _convert_config
from avaturn_live_streamer.worklets.scripted_user import ScriptedUser

_MODEL = "avtrn-1"

AGENT_PROMPT = (
    "You are a voice agent used for latency tests. When the user asks you to count, count from "
    "one to ten at a normal pace, saying only the numbers. When the user interrupts you, stop at "
    "once. Whenever the user asks you to count again, start again from one. Never say anything "
    "else."
)


def _engine_options(args: argparse.Namespace) -> OpenAIEngineOptions | CartesiaEngineOptions:
    values = read_env_files(args.env_file)
    if args.engine == "openai":
        return OpenAIEngineOptions(api_key=openai_key(values), prompt=AGENT_PROMPT)
    missing = [k for k in ("CARTESIA_API_KEY", "CARTESIA_AGENT_ID") if not values.get(k)]
    if missing:
        raise SystemExit(f"missing in the env files: {', '.join(missing)}")
    return CartesiaEngineOptions(
        api_key=values["CARTESIA_API_KEY"], agent_id=values["CARTESIA_AGENT_ID"]
    )


async def main(args: argparse.Namespace) -> list[dict]:
    args.timing = "avtr1"
    durations = _durations(args)
    if args.renderer_url is None:
        avatar, background = "mock", "mock"
    else:
        avatar, background = await _pick_avatar(args)
    renderer_config = RendererConfig(
        avatar_id=RendererAvatarId(avatar),
        background_id=BackgroundId(background),
        pixel_format=PixelFormat.YUV_I420,
        model=_MODEL,
    )
    if args.renderer_url is None:
        _make_frames(_convert_config(renderer_config))
        t = TIMINGS["avtr1"]
        renderer: AbstractRendererClient = MockRendererClient(
            timing_source=fixed_frame_times(t["ttff"] / 1000, t["mtbf"] / 1000)
        )
    else:
        renderer = RendererClient(model=_MODEL, instance_url=args.renderer_url, lb_url=None)
    registry = RendererClientRegistry({_MODEL: renderer})

    _engine_config, engine_run = await build_engine(_engine_options(args), stream_id="measure")
    user = ScriptedUser(
        SpeechBuffer.read(str(args.prompts / "prompt.wav")),
        SpeechBuffer.read(str(args.prompts / "interruption.wav")),
        args.turns,
        seed=args.seed,
    )
    worklets = [
        RenderingWorklet(registry, renderer_config, durations=durations).run,
        run_delayed_event_worklet,
        LatencyWorklet().run,
        engine_run,
        user.run,
    ]
    if args.sink == "null":
        await run_stream(run_null_sink_worklet, *worklets)
        return user.results

    server_pc = RTCPeerConnection()
    peer = LocalRTC(server_pc)
    receiver = await _start_receiver(server_pc)
    if args.video_bitrate:
        peer.fix_video_bitrate(args.video_bitrate)
    try:
        await run_stream(
            LocalRTCWorklet(peer, renderer_config.pixel_format, read_user_audio=False).run,
            *worklets,
        )
    finally:
        await server_pc.close()
        assert receiver.stdin is not None
        receiver.stdin.close()
        await receiver.wait()
    return user.results


def summarize(rows: list[dict]) -> None:
    """Print the replies' terms in the layout of the paper's table, as mean [min, max] in ms.

    The streamer logs the padding together with the constant T_future - T_present (5 ms). The table
    reports T_pad without it, so it is subtracted here. Pred. is the scheduler's sum of the terms,
    which includes T_future; Meas. is the wall-clock total.
    """
    t = TIMINGS["avtr1"]
    carry = (t["future"] - t["present"]) / 1000

    def cell(values: list[float]) -> str:
        return f"{mean(values):6.1f} [{min(values):4.0f}, {max(values):4.0f}]" if values else "--"

    replies = [r for r in rows if r["role"] == "reply" and r["rlc"]]
    columns = {"RLC": replies, "ILC": [r for r in replies if r["ilc"]]}
    others = sum(1 for r in rows if r["role"] != "reply")
    print(f"\n{len(replies)} replies, {len(columns['ILC'])} interrupted; {others} other segments")
    terms = {
        "T_wait": lambda m: m["t_wait"],
        "T_pad": lambda m: m["t_pad"] - carry,
        "T_lead": lambda m: m["t_lead"],
        "Pred.": lambda m: m["t_total"],
        "Meas.": lambda m: m["t_total_wall"],
    }
    print(f"{'':8s}{'RLC':>22s}{'ILC':>22s}")
    for name, term in terms.items():
        line = f"{name:8s}"
        for kind, subset in columns.items():
            if name == "T_pad" and kind == "ILC":
                line += f"{'--':>22s}"  # no padding term on interruption (R4)
                continue
            values = [term(r[kind.lower()]) * 1000 for r in subset]
            line += f"{cell(values):>22s}"
        print(line)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--engine", choices=["openai", "cartesia"])
    p.add_argument("--turns", type=int, default=50)
    p.add_argument(
        "--env-file", type=Path, action="append", help="KEY=VALUE files with the API keys"
    )
    p.add_argument("--prompts", type=Path, default=Path("runs/prompts"))
    p.add_argument("--renderer-url", default=None, help="real renderer service; default: mock")
    p.add_argument("--avatar", default=None)
    p.add_argument("--background", default=None)
    p.add_argument("--sink", choices=["null", "webrtc"], default="webrtc")
    p.add_argument("--video-bitrate", type=int, default=1_500_000, help="webrtc: fixed VP8 bitrate")
    p.add_argument("--seed", type=int, default=0, help="seed of the interruption delays")
    p.add_argument("--out", type=Path, help="JSONL output, one line per played segment")
    p.add_argument(
        "--summarize",
        type=Path,
        nargs="+",
        default=None,
        help="only print the table for these JSONL files from earlier runs, then exit",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    if args.summarize:
        for path in args.summarize:
            print(f"== {path}")
            summarize([json.loads(line) for line in path.read_text().splitlines() if line.strip()])
        raise SystemExit(0)
    if args.out is None or args.engine is None or not args.env_file:
        p.error("--engine, --env-file and --out are required")
    setup_logging(LoggingConfig(level=args.log_level))
    rows = asyncio.run(main(args))
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps({"engine": args.engine, **row}) + "\n")
    summarize(rows)
