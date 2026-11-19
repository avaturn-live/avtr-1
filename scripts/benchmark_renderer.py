#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Measure the renderer's per-chunk timing on this GPU, for the runtime table of the paper.

The measurement is offline, in process, as the table states: one request is one five-frame
(200 ms) chunk, run through Pipeline.process_chunk as the renderer service runs it. The chunks
are consecutive windows of two recordings (speech and listening audio), with the state carried
from chunk to chunk.

Per chunk:
    TTFF             time from the call to the first frame on the host
    request latency  time from the call to the fifth frame on the host
    throughput       200 ms divided by the request latency

Usage:
    python scripts/benchmark_renderer.py --avatar arnold --bg plain_white
"""

import argparse
import time
from pathlib import Path
from statistics import mean, median

import numpy as np
import soundfile as sf
import soxr

from avtr1_renderer import Chunk, Pipeline, RenderOptions

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 6480  # (5 present + 5 future frames) * 640 + 80, see Chunk
STEP_SAMPLES = 3200  # 200 ms: one chunk of five frames


def load_16k(path: Path) -> np.ndarray:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = soxr.resample(audio, rate, SAMPLE_RATE)
    return audio.astype(np.float32)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--avatar", default="arnold")
    p.add_argument("--bg", default="plain_white")
    p.add_argument("--speech", type=Path, default=Path("example/speaker_1.ogg"))
    p.add_argument("--listen", type=Path, default=Path("example/speaker_2.ogg"))
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--chunks", type=int, default=200)
    args = p.parse_args()

    pipeline, registry = Pipeline.from_artifacts(avatar_ids=[args.avatar])
    avatar = registry[args.avatar]
    options = RenderOptions(bg_id=args.bg)
    speech, listen = load_16k(args.speech), load_16k(args.listen)
    n = min(len(speech), len(listen))
    starts = range(0, n - CHUNK_SAMPLES, STEP_SAMPLES)
    if len(starts) < args.warmup + args.chunks:
        raise SystemExit(f"the recordings give only {len(starts)} chunks")

    state = None
    ttff, latency = [], []
    for i, start in enumerate(starts[: args.warmup + args.chunks]):
        chunk = Chunk(
            audio_speech=speech[start : start + CHUNK_SAMPLES].copy(),
            audio_listen=listen[start : start + CHUNK_SAMPLES].copy(),
        )
        t0 = time.perf_counter()
        state, frames = pipeline.process_chunk(avatar, chunk, state, options)
        first = None
        count = 0
        for frame in frames:
            frame.data.tobytes()  # the bytes the service sends; the frame is on the host
            count += 1
            if first is None:
                first = time.perf_counter() - t0
        total = time.perf_counter() - t0
        if count != 5:
            raise SystemExit(f"chunk {i}: {count} frames, expected 5")
        if i >= args.warmup:
            ttff.append(first * 1000)
            latency.append(total * 1000)

    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}; {len(latency)} chunks after {args.warmup} warm-up")
    for name, v in (("TTFF", ttff), ("request latency", latency)):
        print(
            f"{name:16s} mean {mean(v):6.1f}  median {median(v):6.1f}  p99 {percentile(v, 0.99):6.1f}"
            f"  max {max(v):6.1f} ms"
        )
    print(f"real-time throughput (200 ms / mean request latency): {200 / mean(latency):.2f}x")


if __name__ == "__main__":
    main()
