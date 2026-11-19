# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Local WebRTC peer-state object backed by aiortc.

Parallel role to `streamer/rtc.py`'s `RTC` (which wraps Daily): owns the
`RTCPeerConnection`, the outbound video/audio tracks, and the inbound audio
queue for one local browser peer. Constructed *after* SDP negotiation by
the local-stream CLI's signaling handler, then handed to `LocalRTCWorklet`.

PTS on outbound video frames is preserved as-set by the worklet (the worklet
applies the hybrid nominal/wall-clock rule); the track does not re-stamp.
"""

import asyncio
import fractions
from typing import Literal

import av
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection
from aiortc.codecs.vpx import Vp8Encoder
from aiortc.mediastreams import MediaStreamError
from av.frame import Frame as AvFrame

from avaturn_live_streamer import constant
from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer

_LOGGER = get_logger()

_AUDIO_TIME_BASE = fractions.Fraction(1, constant.NATIVE_SPEECH_SAMPLE_RATE)
_AUDIO_SAMPLES_PER_FRAME = constant.NATIVE_SPEECH_SAMPLE_RATE // 50  # 20ms

type ConnectionState = Literal["new", "connecting", "connected", "disconnected", "failed", "closed"]


class _QueuedVideoTrack(MediaStreamTrack):
    """Outbound video track. recv() awaits frames enqueued by the worklet."""

    kind = "video"

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[av.VideoFrame | None] = asyncio.Queue(maxsize=2)

    async def push(self, frame: av.VideoFrame) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(frame)

    async def recv(self) -> AvFrame:
        frame = await self._queue.get()
        if frame is None:
            self.stop()
            raise ConnectionError("video track closed")
        return frame


class _QueuedAudioTrack(MediaStreamTrack):
    """Outbound audio track. recv() pops PCM samples written by the worklet."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[av.AudioFrame | None] = asyncio.Queue(maxsize=64)
        self._pts: int = 0

    async def push(self, samples: np.ndarray) -> None:
        """`samples` is int16 mono shaped (N,) or (1, N)."""
        if samples.ndim == 2:
            samples = samples.reshape(-1)
        # Re-chunk into AUDIO_SAMPLES_PER_FRAME segments so each av.AudioFrame is small.
        for start in range(0, samples.shape[0], _AUDIO_SAMPLES_PER_FRAME):
            chunk = samples[start : start + _AUDIO_SAMPLES_PER_FRAME]
            frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = constant.NATIVE_SPEECH_SAMPLE_RATE
            frame.pts = self._pts
            frame.time_base = _AUDIO_TIME_BASE
            self._pts += chunk.shape[0]
            await self._queue.put(frame)

    async def recv(self) -> AvFrame:
        frame = await self._queue.get()
        if frame is None:
            self.stop()
            raise ConnectionError("audio track closed")
        return frame


class _FixedBitrateVp8Encoder(Vp8Encoder):
    """VP8 encoder that ignores bitrate changes from congestion control.

    aiortc builds a new libvpx encoder each time the target bitrate changes by more than 10 %.
    Building one, with its first keyframe, holds the GIL for 40-100 ms, so the event loop stalls
    for that time. With a fixed bitrate, the encoder is built once, at the first frame.
    """

    def __init__(self, bitrate: int) -> None:
        super().__init__()
        self._fixed_bitrate = bitrate
        self._Vp8Encoder__target_bitrate = bitrate  # read by the encoder options (bufsize)

    @property
    def target_bitrate(self) -> int:
        return self._fixed_bitrate

    @target_bitrate.setter
    def target_bitrate(self, bitrate: int) -> None:
        pass  # Keep the fixed bitrate.


class LocalRTC:
    """Owns the peer connection, outbound tracks, and inbound audio buffer.

    Constructed once SDP negotiation is complete. Handed to `LocalRTCWorklet`,
    which drives `push_video_frame` / `push_audio_chunk` and consumes
    `recv_audio_chunk`. Lifecycle (close) is the responsibility of the CLI app.
    """

    def __init__(self, pc: RTCPeerConnection) -> None:
        self.pc = pc
        self._video_track = _QueuedVideoTrack()
        self._audio_track = _QueuedAudioTrack()
        self._inbound_audio: asyncio.Queue[SpeechBuffer] = asyncio.Queue(maxsize=600)
        self._connection_state: ConnectionState = "new"
        self._state_waiters: list[asyncio.Future[ConnectionState]] = []

        pc.addTrack(self._video_track)
        pc.addTrack(self._audio_track)
        pc.on("connectionstatechange", self._on_state_change)
        pc.on("track", self._on_track)

    def _on_state_change(self) -> None:
        state: ConnectionState = self.pc.connectionState  # pyright: ignore [reportAssignmentType]
        _LOGGER.info("localrtc peer connection state", state=state)
        self._connection_state = state
        waiters = self._state_waiters
        self._state_waiters = []
        for w in waiters:
            if not w.done():
                w.set_result(state)

    def _on_track(self, track: MediaStreamTrack) -> None:
        if track.kind != "audio":
            _LOGGER.debug("localrtc ignoring inbound track", kind=track.kind)
            return
        _LOGGER.info("localrtc inbound audio track attached")
        asyncio.create_task(self._consume_inbound_audio(track))

    async def _consume_inbound_audio(self, track: MediaStreamTrack) -> None:
        target_sr = constant.NATIVE_SPEECH_SAMPLE_RATE
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
        try:
            while True:
                frame = await track.recv()
                if not isinstance(frame, av.AudioFrame):
                    continue
                for out_frame in resampler.resample(frame):
                    arr = out_frame.to_ndarray().reshape(-1).astype(np.int16, copy=False)
                    if arr.size == 0:
                        continue
                    buf = SpeechBuffer(arr, target_sr)
                    if self._inbound_audio.full():
                        _LOGGER.warning("localrtc inbound audio queue full, dropping oldest")
                        try:
                            self._inbound_audio.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await self._inbound_audio.put(buf)
        except (asyncio.CancelledError, ConnectionError):
            raise
        except MediaStreamError:
            _LOGGER.info("localrtc inbound audio track ended")
        except Exception:
            _LOGGER.exception("localrtc inbound audio loop crashed")

    @property
    def connection_state(self) -> ConnectionState:
        return self._connection_state

    async def wait_state_change(self) -> ConnectionState:
        fut: asyncio.Future[ConnectionState] = asyncio.get_event_loop().create_future()
        self._state_waiters.append(fut)
        return await fut

    async def wait_connected(self, timeout: float = 30.0) -> None:
        """Block until the peer connection reaches ``"connected"``.

        Raises ``TimeoutError`` if the connection doesn't establish within
        ``timeout`` seconds, and ``ConnectionError`` if the peer enters a
        terminal failure state ("closed"/"failed") first.
        """
        async with asyncio.timeout(timeout):
            while self._connection_state != "connected":
                if self._connection_state in ("closed", "failed"):
                    raise ConnectionError(
                        f"peer connection entered terminal state {self._connection_state!r} "
                        f"before reaching 'connected'"
                    )
                await self.wait_state_change()

    def fix_video_bitrate(self, bitrate: int) -> None:
        """Send VP8 at a fixed bitrate, without congestion control. For measurement runs only.

        Call after SDP negotiation and before the first video frame. The interactive demo does not
        call it, so it keeps the adaptive bitrate. Raises RuntimeError if VP8 was not negotiated.
        """
        for transceiver in self.pc.getTransceivers():
            if transceiver.sender.track is not self._video_track:
                continue
            codecs = transceiver._codecs  # negotiated codecs, preferred first (aiortc internal)
            if not codecs or codecs[0].mimeType.lower() != "video/vp8":
                raise RuntimeError(f"fix_video_bitrate needs VP8, negotiated: {codecs[:1]}")
            # aiortc creates the encoder at the first frame only when none is set.
            transceiver.sender._RTCRtpSender__encoder = _FixedBitrateVp8Encoder(bitrate)
            _LOGGER.info("localrtc VP8 bitrate fixed", bitrate=bitrate)
            return
        raise RuntimeError("fix_video_bitrate: no video sender")

    async def push_video_frame(self, frame: av.VideoFrame) -> None:
        await self._video_track.push(frame)

    async def push_audio_chunk(self, speech: SpeechBuffer) -> None:
        if speech.is_empty:
            return
        resampled = speech.resample(constant.NATIVE_SPEECH_SAMPLE_RATE)
        samples = np.frombuffer(resampled.to_bytes(), dtype=np.int16)
        await self._audio_track.push(samples)

    async def recv_audio_chunk(self) -> SpeechBuffer:
        return await self._inbound_audio.get()

    async def close(self) -> None:
        await self.pc.close()
