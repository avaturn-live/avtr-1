#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Receiving WebRTC peer for `run_scripted_sweep.py --sink webrtc`, run as a child process.

It has its own process, so its RTP handling does not share the event loop or the GIL with the
streamer. It takes the role of the browser: it offers recvonly video and audio. It does not decode:
the decoder thread of aiortc is replaced by one that counts the encoded frames and drops them.

Signaling on stdin and stdout, one JSON line each:
    1. The receiver writes its offer: {"sdp": ..., "type": "offer"}.
    2. The parent writes the answer: {"sdp": ..., "type": "answer"}.
    3. The parent closes stdin to stop the receiver.
Logs go to stderr only, because stdout carries the signaling.
"""

import asyncio
import json
import queue
import sys
from collections import Counter

import aiortc.rtcrtpreceiver
from aiortc import RTCPeerConnection, RTCSessionDescription

_COUNTS: Counter[str] = Counter()


def _drop_worker(
    loop: asyncio.AbstractEventLoop, input_q: queue.Queue, output_q: asyncio.Queue, *_: object
) -> None:
    """Replacement for aiortc's `decoder_worker`: count encoded frames, do not decode them.

    Newer aiortc versions pass more arguments. They are not needed here.
    """
    while True:
        task = input_q.get()
        if task is None:
            asyncio.run_coroutine_threadsafe(output_q.put(None), loop)
            return
        codec, _encoded_frame = task
        _COUNTS[codec.name] += 1


# RTCRtpReceiver starts its decoder thread with the module-level `decoder_worker`.
aiortc.rtcrtpreceiver.decoder_worker = _drop_worker  # pyright: ignore[reportAttributeAccessIssue]


async def main() -> None:
    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")

    await pc.setLocalDescription(await pc.createOffer())
    offer = pc.localDescription
    sys.stdout.write(json.dumps({"sdp": offer.sdp, "type": offer.type}) + "\n")
    sys.stdout.flush()

    answer = json.loads(await asyncio.to_thread(sys.stdin.readline))
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

    # Run until the parent closes stdin.
    await asyncio.to_thread(sys.stdin.read)
    await pc.close()
    print(f"webrtc receiver: encoded frames received {dict(_COUNTS)}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
