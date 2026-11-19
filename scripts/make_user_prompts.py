#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Make the scripted user's two utterances once, with OpenAI text-to-speech.

Writes 24 kHz mono WAV files, so every turn of a measurement run plays identical audio.

Usage:
    python scripts/make_user_prompts.py --env-file ../avaturn-live-core-backend/local.env --out runs/prompts
"""

import argparse
from pathlib import Path

import numpy as np

from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer

UTTERANCES = {
    "prompt": "Please count from one to ten.",
    "interruption": "Stop. Count again.",
}


def read_env_files(paths: list[Path]) -> dict[str, str]:
    """Read KEY=VALUE lines. Later files win. Values are never printed."""
    values: dict[str, str] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip().removeprefix("export ").strip()] = value.strip().strip("\"'")
    return values


def trim_silence(samples: np.ndarray, rate: int, keep: float = 0.05) -> np.ndarray:
    """Cut the silence that TTS leaves before and after the speech, keeping `keep` seconds."""
    loud = np.flatnonzero(np.abs(samples.astype(np.int32)) > 500)
    if len(loud) == 0:
        return samples
    pad = int(keep * rate)
    return samples[max(0, loud[0] - pad) : loud[-1] + pad]


def openai_key(values: dict[str, str]) -> str:
    for name in ("OPENAI_API_KEY", "OPENAI__API_KEY"):
        if values.get(name):
            return values[name]
    raise SystemExit("no OPENAI_API_KEY or OPENAI__API_KEY in the env files")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--env-file", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, default=Path("runs/prompts"))
    p.add_argument("--voice", default="alloy")
    args = p.parse_args()

    from openai import OpenAI

    client = OpenAI(api_key=openai_key(read_env_files(args.env_file)))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in UTTERANCES.items():
        pcm = client.audio.speech.create(
            model="gpt-4o-mini-tts", voice=args.voice, input=text, response_format="pcm"
        ).read()  # 24 kHz, 16-bit, mono
        samples = trim_silence(np.frombuffer(pcm, dtype=np.int16), 24_000)
        speech = SpeechBuffer(samples.copy(), 24_000)
        path = args.out / f"{name}.wav"
        speech.write(str(path))
        print(f"{path}: {float(speech.duration):.2f} s  ({text!r})")


if __name__ == "__main__":
    main()
