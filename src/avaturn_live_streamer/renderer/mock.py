# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import asyncio
import random
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from math import floor
from typing import AsyncGenerator

import numpy as np
from numpy.typing import NDArray

from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.constant import (
    FRAME_DURATION,
    RENDERER_SPEECH_SAMPLE_RATE,
    VIDEO_FPS,
)
from avaturn_live_streamer.renderer.client import aenumerate
from avaturn_live_streamer.renderer.interface import (
    AbstractRendererClient,
    RenderConfig,
    RendererRequest,
    RenderResponse,
)
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.types import PixelFormat
from avaturn_live_streamer.utils.async_utils import aiotime

_LOGGER = get_logger()


def _rgb_to_i420(image: NDArray) -> NDArray:
    """Convert one (H, W, 3) uint8 RGB image to flat I420 bytes (BT.601, as cv2.COLOR_RGB2YUV_I420).

    numpy only, so the mock renderer runs in the streamer env, which has no OpenCV.
    """
    rgb = image.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = (b - y) * 0.564 + 128.0
    v = (r - y) * 0.713 + 128.0
    # 2x2 chroma subsampling by averaging.
    h, w = y.shape
    u = u.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    v = v.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    planes = [np.clip(np.rint(p), 0, 255).astype(np.uint8).ravel() for p in (y, u, v)]
    return np.concatenate(planes)


@lru_cache(maxsize=1)
def _make_frames(config: RenderConfig) -> NDArray:
    timestamps = np.linspace(0, 4, num=100, endpoint=False)
    num_frames = len(timestamps)

    # Shape: (num_frames, height, width, 3)
    images = np.zeros((num_frames, config.height, config.width, 3), dtype=np.uint8)

    # Vectorized red value calculation for all timestamps
    red_values = np.round((np.sin(timestamps * 3) + 1) / 2 * 255).astype(np.uint8)
    images[:, :, :, 0] = red_values[:, np.newaxis, np.newaxis]

    if config.pixel_format == PixelFormat.YUV_I420_STACKED_ALPHA:
        images = np.concatenate((images, np.empty_like(images)), axis=1)
        alpha_gradient = np.linspace(0, 255, config.width, dtype=np.uint8)
        images[:, config.height :, :, :] = alpha_gradient[np.newaxis, np.newaxis, :, np.newaxis]
    elif config.pixel_format == PixelFormat.YUV_I420:
        pass
    else:
        raise ValueError(f"Unknown pixel format: {config.pixel_format}")

    return np.array([_rgb_to_i420(images[i]) for i in range(num_frames)])


def _mock_render(request: RendererRequest) -> RenderResponse:
    duration = SpeechBuffer.from_bytes(request.current_chunk, RENDERER_SPEECH_SAMPLE_RATE).duration

    user_current_chunk_duration = SpeechBuffer.from_bytes(
        request.current_chunk_listen, RENDERER_SPEECH_SAMPLE_RATE
    ).duration
    user_future_chunk_duration = SpeechBuffer.from_bytes(
        request.future_chunk_listen, RENDERER_SPEECH_SAMPLE_RATE
    ).duration
    num_frames = floor(duration * VIDEO_FPS)

    # Always generate state for continuity between render calls
    if request.state is None:
        state = b"0"
    else:
        state = str(int(request.state.decode()) + 1).encode()

    frames = _make_frames(request.config)

    state_future = asyncio.get_running_loop().create_future()

    async def stream_frames():
        _LOGGER.debug(
            f"Mock render client start: {num_frames=} {duration=:.3f} {request.timestamp_global=:.3f} {user_current_chunk_duration=:.3f} {user_future_chunk_duration=:.3f}"
        )
        for i in range(num_frames):
            frame_idx = int(request.timestamp_global / FRAME_DURATION + i) % frames.shape[0]
            yield frames[frame_idx].tobytes()
        _LOGGER.debug(f"Mock render client done: generated {num_frames} frames")

        state_future.set_result(state)

    return RenderResponse(stream_frames(), num_frames, state_future)


def const_frame_time_source(frame_number: int) -> float:
    ttff = 0.05
    mtbf = 0.017
    return ttff if frame_number == 0 else mtbf


def fixed_frame_times(ttff: float, mtbf: float) -> Callable[[int], float]:
    """Timing source: the first frame after `ttff` seconds, each later frame `mtbf` seconds later."""

    def source(frame_number: int) -> float:
        return ttff if frame_number == 0 else mtbf

    return source


def random_frame_time_source(frame_number: int) -> float:
    return random.gauss(0.3, sigma=0.1) if frame_number == 0 else random.gauss(0.02, sigma=0.005)


class MockRendererClient(AbstractRendererClient):
    def __init__(
        self,
        request_validator: Callable[[RendererRequest], None] | None = None,
        timing_source: Callable[[int], float] = const_frame_time_source,
    ):
        self.timing_source = timing_source
        self.request_validator = request_validator

    async def render(self, request: RendererRequest) -> RenderResponse:
        return _mock_render(request)

    async def generate_no_cm(self, request: RendererRequest) -> RenderResponse:
        if self.request_validator:
            self.request_validator(request)

        resp = await self.render(request)

        async def _wrap_generator(start):
            next_frame_at = start
            async for i, frame in aenumerate(resp.frame_generator):
                next_frame_at += self.timing_source(i)
                await asyncio.sleep(next_frame_at - aiotime())
                yield frame

        return RenderResponse(_wrap_generator(aiotime()), resp.num_frames, resp.state)

    @asynccontextmanager
    async def generate(self, request: RendererRequest) -> AsyncGenerator[RenderResponse, None]:
        yield await self.generate_no_cm(request)


def get_frame_size_bytes(h: int, w: int, pixel_format: PixelFormat):
    if pixel_format.is_yuv:
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"Height and width must be even, got {h}")

    if pixel_format == PixelFormat.YUV_I420:
        return h * w * 3 // 2
    elif pixel_format == PixelFormat.YUV_I420_STACKED_ALPHA:
        return h * w * 3
    else:
        raise ValueError(f"Unknown pixel format: {pixel_format}")  # pyright: ignore[reportUnreachable]
