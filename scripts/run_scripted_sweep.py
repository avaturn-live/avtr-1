#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Controlled latency sweep: scripted voice agent, mock renderer, no transport.

The run uses the real streamer worklets (rendering loop, scheduler, delayed playback events,
latency logging) with three stand-ins: the scripted agent in place of a voice agent, the mock
renderer in place of the renderer service, and a null frame sink in place of WebRTC.

The default grid has 120 turns: 10 arrival phases x 4 padding levels x 3 later-fragment sizes.
A padding level p in [0, 1] sets the first fragment to T_present - p * (T_present - T_wait), the
largest T_pad that the real-time condition allows, scaled by p. The first fragment is at least
T_wait + AFTER_ADMIT, so p = 1 gives T_pad = T_present - T_wait - AFTER_ADMIT. The interruption phase takes the same 10 values, rotated, so each phase occurs with every
fragment configuration. Each turn writes one JSON line with its configuration, the prediction and
the measured terms.

Usage:
    pixi run -e streamer python scripts/run_scripted_sweep.py --out sweep.jsonl
    # The sets for the paper figure, then the figure and the numbers for the text:
    pixi run -e streamer python scripts/run_scripted_sweep.py --plan figure --out figure.jsonl
    python scripts/sweep_figure.py figure.jsonl --tikz avtr1-latency-sweep-tikz.tex
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import itertools
import json
import sys
import threading
import time
import traceback
from collections import Counter
from contextlib import asynccontextmanager
from fractions import Fraction
from math import ceil
from pathlib import Path
from statistics import mean

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription

from avaturn_live_streamer.config import RendererConfig
from avaturn_live_streamer.conversation_engines.scripted import (
    AFTER_ADMIT,
    ScriptedAgent,
    ScriptedTurn,
)
from avaturn_live_streamer.core.logging_config import LoggingConfig
from avaturn_live_streamer.core.logs import setup_logging
from avaturn_live_streamer.localrtc import LocalRTC, LocalRTCWorklet
from avaturn_live_streamer.renderer.client import RendererClient
from avaturn_live_streamer.renderer.interface import AbstractRendererClient
from avaturn_live_streamer.renderer.mock import (
    MockRendererClient,
    _make_frames,
    fixed_frame_times,
)
from avaturn_live_streamer.renderer.models import RenderingModelDurations
from avaturn_live_streamer.renderer.registry import RendererClientRegistry
from avaturn_live_streamer.runner import run_stream
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.types import BackgroundId, PixelFormat, RendererAvatarId
from avaturn_live_streamer.worklets.delayed_event import run_delayed_event_worklet
from avaturn_live_streamer.worklets.latency import LatencyWorklet
from avaturn_live_streamer.worklets.null_sink import run_null_sink_worklet
from avaturn_live_streamer.worklets.rendering import RenderingWorklet, _convert_config

_MODEL = "avtrn-1"

# Mock renderer timings, in ms. present = N * 40 ms. The ttff and mtbf values of the two other
# generators are our choice: their papers do not state them. Each lead covers its ttff.
TIMINGS: dict[str, dict[str, int]] = {
    # XXXX-1: N = 5. ttff and mtbf as the default mock (renderer/mock.py). lead 60 ms covers the
    # TTFF measured on an RTX 4090 (p50 45 ms, max 56 ms); the streamer default stays 100 ms.
    "avtr1": {"present": 200, "future": 205, "lead": 60, "ttff": 50, "mtbf": 17},
    # Frame-wise, as DyStream: N = 1, T_future = 60 ms.
    "framewise": {"present": 40, "future": 60, "lead": 40, "ttff": 35, "mtbf": 20},
    # Block-wise, as Avatar Forcing: N = 10, no future window.
    "blockwise": {"present": 400, "future": 0, "lead": 150, "ttff": 100, "mtbf": 20},
}


def _durations(args: argparse.Namespace) -> RenderingModelDurations:
    t = TIMINGS[args.timing]
    lead_ms = getattr(args, "lead_ms", None) or t["lead"]
    return RenderingModelDurations(
        present=Fraction(t["present"], 1000),
        future=Fraction(t["future"], 1000),
        lead=Fraction(lead_ms, 1000),
    )


class _TTFFTracer(AbstractRendererClient):
    """Diagnostic wrapper: record, per request, the time from the request to its first frame."""

    def __init__(self, inner: AbstractRendererClient) -> None:
        self._inner = inner
        self.ttff: list[float] = []

    @asynccontextmanager
    async def generate(self, request):
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        async with self._inner.generate(request) as response:
            frames = response.frame_generator

            async def traced():
                first = True
                async for frame in frames:
                    if first:
                        self.ttff.append(loop.time() - t0)
                        first = False
                    yield frame

            response.frame_generator = traced()
            yield response

    def report(self) -> None:
        if not self.ttff:
            return
        ms = sorted(v * 1000 for v in self.ttff[5:])  # skip the warm-up requests
        n = len(ms)
        q = {p: ms[min(n - 1, int(p * n))] for p in (0.5, 0.9, 0.99, 0.999)}
        print(
            f"TTFF (ms), {n} requests: p50 {q[0.5]:.1f}, p90 {q[0.9]:.1f}, p99 {q[0.99]:.1f}, "
            f"p99.9 {q[0.999]:.1f}, max {ms[-1]:.1f}"
        )


def _ms_list(text: str) -> list[Fraction]:
    return [Fraction(int(v), 1000) for v in text.split(",") if v.strip()]


def build_bounds_turns(args: argparse.Namespace) -> list[ScriptedTurn]:
    """20 turns at the floor of the bounds and 20 at the ceiling, for one renderer timing.

    floor: the reply and the interruption arrive 5 ms before a step (T_wait = 5 ms), and the reply
        has at least T_present of speech (T_pad = 0). Predicted: T_lead + T_future + 5 ms.
    ceiling: the interruption arrives 5 ms after a step (T_wait = T_present - 5 ms). The replies
        cover T_wait from 5 ms to T_present - 5 ms, each with only R = T_wait + 5 ms of speech, so
        T_wait + T_pad = T_present - 5 ms in every turn. Predicted, for every phase:
        T_lead + T_future + T_present - 5 ms.
    The margins keep each arrival off the exact step time. Before a step (floor) the agent's event
    needs one more event-loop pass to reach the scheduler, and epoll rounds its timeout up to whole
    milliseconds, so the floor margin must exceed about 1 ms. After a step (ceiling) the loop fires
    the step's timer first, so any positive margin is safe.
    """
    d = _durations(args)
    tp = d.present
    floor_m, ceil_m = Fraction(args.floor_margin_ms) / 1000, Fraction(args.ceiling_margin_ms) / 1000
    after_admit = Fraction(args.after_admit_ms) / 1000
    # Interrupt after the reply's speech has started: it starts at most T_future + T_present
    # after the admitting step's present window starts.
    steps = max(4, ceil((d.future + 2 * tp) / tp))
    common = {
        "fragment": Fraction(80, 1000),
        "duration": max(Fraction(2), (steps + 4) * tp),
        "rate": None if args.rate == "burst" else Fraction(args.rate),
        "interrupt_after_steps": steps,
    }
    turns = list[ScriptedTurn]()
    for _ in range(args.n_bounds):
        turns.append(
            ScriptedTurn(
                t_wait=floor_m,
                first_fragment=max(tp, floor_m + after_admit),
                interrupt_t_wait=floor_m,
                label="floor",
                **common,
            )
        )
    for i in range(args.n_bounds):
        t_wait = floor_m + (tp - floor_m - ceil_m) * i / max(1, args.n_bounds - 1)
        t_wait = Fraction(round(t_wait * 2000), 2000)  # whole 0.5 ms, so whole samples
        turns.append(
            ScriptedTurn(
                t_wait=t_wait,
                first_fragment=t_wait + after_admit,
                interrupt_t_wait=tp - ceil_m,
                label="ceiling",
                **common,
            )
        )
    return turns


def build_figure_turns(args: argparse.Namespace) -> list[ScriptedTurn]:
    """Turns for the three panels of the paper's sweep figure.

    Set A (panels a and c): T_pad = 0, 13 values of T_wait evenly from 5 to 195 ms, each with
    later fragments of 20, 80 and 200 ms: 3 turns per value. The interruption phase takes the same
    13 values, shifted against the arrival phase, so each phase also occurs once per fragment size.
    Set B (panel b): T_wait = 20 ms, 14 first-fragment sizes from 30 to 400 ms, including 200 ms
    (T_present, where the padding ends), 3 turns each. 30 ms is the smallest first fragment that
    the real-time assumption allows (T_wait + AFTER_ADMIT).
    """
    tp = _durations(args).present
    rate = None if args.rate == "burst" else Fraction(args.rate)
    common = {
        "duration": Fraction(args.duration_ms, 1000),
        "rate": rate,
        "interrupt_after_steps": args.interrupt_after_steps,
    }
    floor_m, ceil_m = Fraction(args.floor_margin_ms) / 1000, Fraction(args.ceiling_margin_ms) / 1000
    after_admit = Fraction(args.after_admit_ms) / 1000
    span = tp - floor_m - ceil_m
    # 13 phases from the floor margin to T_present minus the ceiling margin, in whole 0.5 ms.
    phases = [Fraction(round((floor_m + span * i / 12) * 2000), 2000) for i in range(13)]
    turns = list[ScriptedTurn]()
    for f_idx, fragment_ms in enumerate((20, 80, 200)):
        for t_idx, t_wait in enumerate(phases):
            turns.append(
                ScriptedTurn(
                    t_wait=t_wait,
                    first_fragment=max(tp, t_wait + after_admit),  # R >= T_present: T_pad = 0
                    fragment=Fraction(fragment_ms, 1000),
                    interrupt_t_wait=phases[(t_idx + 5 * f_idx + 7) % len(phases)],
                    label="A",
                    **common,
                )
            )
    first_fragments_ms = [30, 60, 90, 120, 150, 180, 200, 220, 250, 280, 310, 340, 370, 400]
    for _ in range(3):
        for r_ms in first_fragments_ms:
            turns.append(
                ScriptedTurn(
                    t_wait=Fraction(20, 1000),
                    first_fragment=Fraction(r_ms, 1000),
                    fragment=Fraction(80, 1000),
                    interrupt_t_wait=Fraction(100, 1000),
                    label="B",
                    **common,
                )
            )
    return turns * args.repeats


def build_turns(args: argparse.Namespace) -> list[ScriptedTurn]:
    if args.plan == "figure":
        return build_figure_turns(args)
    if args.plan == "bounds":
        return build_bounds_turns(args)
    rate = None if args.rate == "burst" else Fraction(args.rate)
    t_waits = _ms_list(args.t_wait)
    turns = list[ScriptedTurn]()
    tp = _durations(args).present
    pad_levels = [Fraction(v) for v in args.pad_levels.split(",")]
    grid = itertools.product(t_waits, pad_levels, _ms_list(args.fragment))
    for i, (t_wait, level, fragment) in enumerate(grid):
        first = max(tp - level * (tp - t_wait), t_wait + AFTER_ADMIT)
        first = Fraction(round(first * 1000), 1000)  # whole ms, so whole samples
        # Rotate the interruption phase against the arrival phase.
        interrupt_t_wait = t_waits[(i // len(t_waits) + i) % len(t_waits)]
        turns.append(
            ScriptedTurn(
                t_wait=t_wait,
                first_fragment=first,
                fragment=fragment,
                duration=Fraction(args.duration_ms, 1000),
                rate=rate,
                interrupt_after_steps=args.interrupt_after_steps,
                interrupt_t_wait=interrupt_t_wait,
            )
        )
    return turns * args.repeats


def summarize(rows: list[dict]) -> None:
    def errors(kind: str) -> list[float]:
        return [
            (r[kind]["t_total_wall"] - r["predicted"][kind]) * 1000
            for r in rows
            if r.get(kind) is not None and kind in r["predicted"]
        ]

    print(f"\nturns: {len(rows)}")
    for kind in ("rlc", "ilc"):
        errs = errors(kind)
        if not errs:
            continue
        print(
            f"{kind.upper()} wall - predicted (ms): mean {mean(errs):.2f}, "
            f"min {min(errs):.2f}, max {max(errs):.2f}"
        )
        # Prediction from the measured arrival, not the planned one (removes the agent's own
        # lateness): the residual is the wall total minus the media total.
        resid = [(r[kind]["t_total_wall"] - r[kind]["t_total"]) * 1000 for r in rows if r.get(kind)]
        print(
            f"{kind.upper()} wall - media (ms): mean {mean(resid):.2f}, "
            f"min {min(resid):.2f}, max {max(resid):.2f}"
        )
    walls = [r["rlc"]["t_total_wall"] * 1000 for r in rows if r.get("rlc")]
    if walls:
        print(f"RLC wall (ms): min {min(walls):.1f}, max {max(walls):.1f}")
    inconsistent = sum(
        1 for r in rows for k in ("rlc", "ilc") if r.get(k) and not r[k]["consistent"]
    )
    print(f"turns with inconsistent terms: {inconsistent}")


_heartbeat = [0.0]


_stalls: list[
    tuple[float, float, float, int]
] = []  # (wall time, lag, loop-thread CPU, ctx switches)
_gc_passes: list[tuple[float, int, float]] = []  # (wall time, generation, duration)


def _ctx_switches() -> int:
    """Involuntary context switches of the calling thread (Linux)."""
    try:
        status = Path(f"/proc/self/task/{threading.get_native_id()}/status").read_text()
    except OSError:
        return -1
    return int(status.split("nonvoluntary_ctxt_switches:")[1].split()[0])


def _gc_callback(phase: str, info: dict) -> None:
    if phase == "start":
        _gc_start[0] = time.perf_counter()
    else:
        duration = time.perf_counter() - _gc_start[0]
        if duration > 0.001:
            _gc_passes.append((time.monotonic(), info["generation"], duration))


_gc_start = [0.0]


async def _probe_loop_lag(lags: list[float]) -> None:
    """Diagnostic: sleep 1 ms in a loop and record how late each wake-up is.

    For each lag over 5 ms, also record the CPU time the loop thread used meanwhile, and its
    involuntary context switches. CPU close to the lag: the loop thread itself was busy. CPU close
    to zero: it waited (for the GIL, or for the OS to schedule it).
    """
    loop = asyncio.get_running_loop()
    gc.callbacks.append(_gc_callback)
    _trace_vp8_encoder()
    while True:
        t0, cpu0, cs0 = loop.time(), time.thread_time(), _ctx_switches()
        _heartbeat[0] = time.monotonic()
        await asyncio.sleep(0.001)
        lag = loop.time() - t0 - 0.001
        lags.append(lag)
        if lag > 0.005:
            _stalls.append(
                (time.monotonic(), lag, time.thread_time() - cpu0, _ctx_switches() - cs0)
            )


def _sample_stalls(loop_thread: int, threshold: float, stop: threading.Event, out: list) -> None:
    """Diagnostic thread: while the loop heartbeat is older than `threshold`, record the loop
    thread's stack once per millisecond. Each record is (stall number, stack as text)."""
    stall, in_stall = 0, False
    while not stop.wait(0.001):
        if _heartbeat[0] == 0.0 or time.monotonic() - _heartbeat[0] < threshold:
            in_stall = False
            continue
        if not in_stall:
            stall, in_stall = stall + 1, True
            # The thread that held the GIL during the stall released it just now: record every
            # thread, with the stall length so far.
            names = {t.ident: t.name for t in threading.enumerate()}
            stall_ms = (time.monotonic() - _heartbeat[0]) * 1000
            parts = [f"stall length so far: {stall_ms:.1f} ms"]
            for ident, frame in sys._current_frames().items():
                if ident == threading.get_ident():
                    continue
                tag = "LOOP " if ident == loop_thread else ""
                parts.append(f"[{tag}{names.get(ident, ident)}]")
                parts.append("".join(traceback.format_stack(frame, limit=6)))
            out.append((stall, "\n".join(parts)))


def _report_stalls(samples: list) -> None:
    if not samples:
        print("stall sampler: no stalls")
        return
    by_stall: dict[int, list[str]] = {}
    for stall, stack in samples:
        by_stall.setdefault(stall, []).append(stack)
    longest = sorted(by_stall.items(), key=lambda kv: -len(kv[1]))[:6]
    print(f"stall sampler: {len(by_stall)} stalls, {len(samples)} samples. Longest:")
    for stall, stacks in longest:
        top = Counter(stacks).most_common(1)[0][0]
        print(f"--- stall {stall}: {len(stacks)} samples (~ms); most common stack:\n{top}")


_vp8_new: list[tuple[float, float, int]] = []  # (wall time at end, duration, target bitrate)


def _trace_vp8_encoder() -> None:
    """Diagnostic: record each Vp8Encoder.encode call that creates a new libvpx encoder."""
    import aiortc.codecs.vpx as vpx

    original = vpx.Vp8Encoder.encode

    def encode(self, frame, force_keyframe=False):
        bitrate_change = self.codec is not None and (
            abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate > 0.1
        )
        new = self.codec is None or bitrate_change
        t0 = time.perf_counter()
        result = original(self, frame, force_keyframe)
        if new:
            _vp8_new.append((time.monotonic(), time.perf_counter() - t0, self.target_bitrate))
        return result

    vpx.Vp8Encoder.encode = encode


def _report_loop_lag(lags: list[float]) -> None:
    if not lags:
        return
    if _vp8_new:
        print(
            f"new VP8 encoders: {len(_vp8_new)}; encode call with a new encoder (ms): "
            f"{sorted(round(d * 1000, 1) for _, d, _ in _vp8_new)[-8:]} (largest 8)"
        )
        for when, lag, *_ in sorted(_stalls, key=lambda x: -x[1])[:6]:
            near = [
                f"{(t - when) * 1000:+.0f} ms ({d * 1000:.0f} ms call, {b // 1000} kbps)"
                for t, d, b in _vp8_new
                if when - lag - 0.1 <= t <= when + 0.1
            ]
            print(f"stall {lag * 1000:5.1f} ms: new encoder within 100 ms: {near or 'none'}")
    for when, lag, cpu, cs in sorted(_stalls, key=lambda x: -x[1])[:10]:
        gcs = [
            f"gen{g} {d * 1000:.1f} ms" for t, g, d in _gc_passes if when - lag - 0.002 <= t <= when
        ]
        print(
            f"stall {lag * 1000:6.1f} ms: loop-thread CPU {cpu * 1000:6.1f} ms, "
            f"involuntary ctx switches {cs}, GC during it: {gcs or 'none'}"
        )
    print(
        f"GC passes over 1 ms: {len(_gc_passes)}, longest {max((d for *_, d in _gc_passes), default=0) * 1000:.1f} ms"
    )
    ms = sorted(v * 1000 for v in lags)
    n = len(ms)
    pct = {p: ms[min(n - 1, int(p * n))] for p in (0.5, 0.99, 0.999)}
    print(
        f"loop lag (ms), {n} wake-ups: p50 {pct[0.5]:.2f}, p99 {pct[0.99]:.2f}, "
        f"p99.9 {pct[0.999]:.2f}, max {ms[-1]:.2f}; over 3 ms: {sum(v > 3 for v in ms)}, "
        f"over 5 ms: {sum(v > 5 for v in ms)}, over 10 ms: {sum(v > 10 for v in ms)}"
    )


async def main(args: argparse.Namespace) -> list[dict]:
    if args.switch_interval_ms is not None:
        sys.setswitchinterval(args.switch_interval_ms / 1000)
    lags: list[float] = []
    probe = asyncio.create_task(_probe_loop_lag(lags)) if args.probe_loop_lag else None
    stop, samples = threading.Event(), []
    if args.sample_stalls_ms is not None:
        threading.Thread(
            target=_sample_stalls,
            args=(threading.get_ident(), args.sample_stalls_ms / 1000, stop, samples),
            daemon=True,
        ).start()
    try:
        return await _main(args)
    finally:
        stop.set()
        if probe is not None:
            probe.cancel()
            _report_loop_lag(lags)
        if args.sample_stalls_ms is not None:
            _report_stalls(samples)


async def _main(args: argparse.Namespace) -> list[dict]:
    if args.log_slow_callbacks_ms is not None:
        # Diagnostic: asyncio logs each callback that holds the event loop longer than this.
        loop = asyncio.get_running_loop()
        loop.set_debug(True)
        loop.slow_callback_duration = args.log_slow_callbacks_ms / 1000
    durations = _durations(args)
    if args.renderer_url is not None and args.timing != "avtr1":
        raise SystemExit("--timing other than avtr1 needs the mock renderer")
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
        # Build the mock frames before the stream starts. The first build takes seconds and blocks
        # the event loop, which would otherwise delay the bus start.
        _make_frames(_convert_config(renderer_config))
        t = TIMINGS[args.timing]
        renderer: AbstractRendererClient = MockRendererClient(
            timing_source=fixed_frame_times(t["ttff"] / 1000, t["mtbf"] / 1000)
        )
    else:
        # The real renderer service, as in the interactive demo (mode "single").
        renderer = RendererClient(model=_MODEL, instance_url=args.renderer_url, lb_url=None)
    tracer = None
    if args.trace_ttff:
        tracer = renderer = _TTFFTracer(renderer)
    registry = RendererClientRegistry({_MODEL: renderer})
    try:
        return await _run(args, durations, renderer_config, registry)
    finally:
        if tracer is not None:
            tracer.report()


async def _run(args, durations, renderer_config, registry) -> list[dict]:
    speech = None if args.speech == "silence" else SpeechBuffer.read(str(args.speech))
    agent = ScriptedAgent(
        build_turns(args),
        durations,
        gap_steps=args.gap_steps,
        speech=speech,
        after_admit=Fraction(args.after_admit_ms) / 1000,
    )

    if args.sink == "null":
        await run_stream(
            RenderingWorklet(registry, renderer_config, durations=durations).run,
            run_null_sink_worklet,
            run_delayed_event_worklet,
            LatencyWorklet().run,
            agent.run,
        )
        return agent.results

    server_pc = RTCPeerConnection()
    peer = LocalRTC(server_pc)
    if args.mute == "video":  # diagnostic: aiortc gets no video frames, so it encodes no VP8

        async def _drop_video(frame: object) -> None:
            return None

        peer.push_video_frame = _drop_video  # type: ignore[method-assign]
    elif args.mute == "audio":  # diagnostic: aiortc gets no audio, so it encodes no Opus

        async def _drop_audio(speech: object) -> None:
            return None

        peer.push_audio_chunk = _drop_audio  # type: ignore[method-assign]
    receiver = await _start_receiver(server_pc)
    if args.video_bitrate:
        # Without this, aiortc rebuilds the VP8 encoder on bitrate changes, which stalls the event
        # loop for 40-100 ms (see LocalRTC.fix_video_bitrate). The interactive demo keeps it adaptive.
        peer.fix_video_bitrate(args.video_bitrate)
    try:
        await run_stream(
            RenderingWorklet(registry, renderer_config, durations=durations).run,
            LocalRTCWorklet(peer, renderer_config.pixel_format, read_user_audio=False).run,
            run_delayed_event_worklet,
            LatencyWorklet().run,
            agent.run,
        )
    finally:
        await server_pc.close()
        assert receiver.stdin is not None
        receiver.stdin.close()
        await receiver.wait()
    return agent.results


async def _pick_avatar(args: argparse.Namespace) -> tuple[str, str]:
    """Return the avatar and background ids: from the options, else the first ones listed."""
    async with httpx.AsyncClient(timeout=10.0) as http:
        r = await http.get(f"{args.renderer_url.rstrip('/')}/avatars")
        r.raise_for_status()
        listed = r.json()
    avatar = args.avatar or listed["avatars"][0]
    background = args.background or listed["backgrounds"][0]
    print(f"renderer {args.renderer_url}: avatar {avatar}, background {background}")
    return avatar, background


async def _start_receiver(server_pc: RTCPeerConnection) -> asyncio.subprocess.Process:
    """Start the receiving peer (`webrtc_receiver.py`) in a child process and connect to it.

    A child process keeps the receiver's decoding off the streamer's event loop and GIL. The
    receiver offers, and the streamer answers, as with the browser UI. ICE uses host candidates.
    """
    receiver = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).with_name("webrtc_receiver.py")),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    assert receiver.stdin is not None and receiver.stdout is not None
    offer = json.loads(await receiver.stdout.readline())
    await server_pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
    )
    await server_pc.setLocalDescription(await server_pc.createAnswer())
    answer = server_pc.localDescription
    receiver.stdin.write((json.dumps({"sdp": answer.sdp, "type": answer.type}) + "\n").encode())
    await receiver.stdin.drain()
    return receiver


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--t-wait",
        default="10,30,50,70,90,110,130,150,170,190",
        help="arrival phases T_wait, ms, comma-separated, each in (0, T_present)",
    )
    p.add_argument(
        "--pad-levels", default="0,1/3,2/3,1", help="padding levels in [0, 1], see the docstring"
    )
    p.add_argument("--fragment", default="20,80,200", help="later-fragment sizes, ms")
    p.add_argument(
        "--rate", default="2", help="pace of later fragments as a multiple of real time, or 'burst'"
    )
    p.add_argument("--duration-ms", type=int, default=2000, help="speech per turn, ms")
    p.add_argument("--interrupt-after-steps", type=int, default=4)
    p.add_argument("--gap-steps", type=int, default=3)
    p.add_argument(
        "--plan",
        choices=["grid", "figure", "bounds"],
        default="grid",
        help="grid: the product of the options below; figure: the sets for the paper figure; "
        "bounds: 20 floor and 20 ceiling turns for --timing",
    )
    p.add_argument(
        "--timing",
        choices=sorted(TIMINGS),
        default="avtr1",
        help="renderer timing (mock renderer); see TIMINGS",
    )
    p.add_argument("--n-bounds", type=int, default=20, help="turns per set for --plan bounds")
    p.add_argument(
        "--floor-margin-ms",
        type=float,
        default=2.0,
        help="figure and bounds plans: smallest T_wait, before a step (must exceed about 1 ms)",
    )
    p.add_argument(
        "--ceiling-margin-ms",
        type=float,
        default=0.5,
        help="figure and bounds plans: T_present minus the largest T_wait, after a step",
    )
    p.add_argument(
        "--after-admit-ms",
        type=float,
        default=None,
        help="later fragments start this long after the admitting step "
        "(default: the ceiling margin for the figure and bounds plans, 10 otherwise)",
    )
    p.add_argument("--repeats", type=int, default=1, help="repeat the whole grid this many times")
    p.add_argument(
        "--speech",
        default=str(Path(__file__).resolve().parent.parent / "example" / "speaker_1.ogg"),
        help="recording the agent replays, or 'silence'",
    )
    p.add_argument(
        "--lead-ms",
        type=int,
        default=None,
        help="override T_lead of the timing, in ms (the agent and the streamer both use it)",
    )
    p.add_argument(
        "--trace-ttff",
        action="store_true",
        help="diagnostic: record the time from each renderer request to its first frame",
    )
    p.add_argument(
        "--renderer-url",
        default=None,
        help="URL of a running renderer service, e.g. http://localhost:8000. Default: mock renderer",
    )
    p.add_argument("--avatar", default=None, help="avatar id (real renderer only)")
    p.add_argument("--background", default=None, help="background id (real renderer only)")
    p.add_argument(
        "--sink",
        choices=["null", "webrtc"],
        default="null",
        help="null: no transport; webrtc: LocalRTCWorklet to an aiortc receiver in a child process",
    )
    p.add_argument("--out", type=Path, required=True, help="JSONL output, one line per turn")
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--video-bitrate",
        type=int,
        default=1_500_000,
        help="webrtc sink: fixed VP8 bitrate in bit/s (default: aiortc's maximum); 0: adaptive",
    )
    p.add_argument(
        "--mute",
        choices=["video", "audio"],
        default=None,
        help="diagnostic (webrtc sink): do not give aiortc this track's frames",
    )
    p.add_argument(
        "--probe-loop-lag", action="store_true", help="diagnostic: measure event-loop lag"
    )
    p.add_argument(
        "--sample-stalls-ms",
        type=float,
        default=None,
        help="diagnostic: record the loop stack during stalls longer than this (needs --probe-loop-lag)",
    )
    p.add_argument(
        "--switch-interval-ms",
        type=float,
        default=None,
        help="diagnostic: GIL switch interval (Python default: 5 ms)",
    )
    p.add_argument(
        "--log-slow-callbacks-ms",
        type=float,
        default=None,
        help="diagnostic: log event-loop callbacks longer than this (asyncio debug mode)",
    )
    args = p.parse_args()

    if args.after_admit_ms is None:
        args.after_admit_ms = args.ceiling_margin_ms if args.plan in ("bounds", "figure") else 10
    setup_logging(LoggingConfig(level=args.log_level))
    rows = asyncio.run(main(args))
    for row in rows:
        row["timing"] = {"name": args.timing, **TIMINGS[args.timing]}
        if args.lead_ms:
            row["timing"]["lead"] = args.lead_ms
        row["timing"]["renderer"] = args.renderer_url or "mock"
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summarize(rows)
