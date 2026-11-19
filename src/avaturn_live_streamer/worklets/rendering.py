# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Speech-to-video worklet with event bus integration."""

from asyncio import TaskGroup
from fractions import Fraction

from asyncstdlib import enumerate as aenumerate
from attrs import Factory, define
from opentelemetry import baggage, metrics

from avaturn_live_streamer import constant
from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.config import RendererConfig
from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    DiscardAvatarSpeechBuffer,
    Frame,
    ScheduledEvent,
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    SegmentPlaybackCancelled,
    SegmentPlaybackCompleted,
    SegmentPlaybackInterrupted,
    SegmentPlaybackStarted,
    Shutdown,
    TurnLatencies,
    UserSpeechReceived,
    UserSpeechStreamEnd,
    UserSpeechStreamStart,
    VideoFrameGenerated,
)
from avaturn_live_streamer.management.types import SegmentId, SegmentMetadata
from avaturn_live_streamer.renderer import RendererClientRegistry
from avaturn_live_streamer.renderer.interface import (
    AbstractRendererClient,
    RenderConfig,
    RendererRequest,
)
from avaturn_live_streamer.renderer.models import (
    RenderingModelDurations,
    get_model_default_extra_params,
    get_model_durations,
)
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.speech.speech_scheduler import (
    SegmentCancelled,
    SegmentInterrupted,
    SegmentLatencies,
    SpeechScheduler,
    TimestampedEvent,
)
from avaturn_live_streamer.speech.speech_scheduler import (
    SegmentEnded as SchedulerSegmentEnded,
)
from avaturn_live_streamer.speech.speech_scheduler import (
    SegmentStarted as SchedulerSegmentStarted,
)
from avaturn_live_streamer.utils.async_utils import run_in_thread
from avaturn_live_streamer.utils.exceptions import async_log_entry_exit

_LOGGER = get_logger()
_METER = metrics.get_meter(__name__)


def _to_wire_latencies(latencies: SegmentLatencies | None) -> TurnLatencies | None:
    if latencies is None:
        return None
    return TurnLatencies(
        kind=latencies.kind,
        t_wait=float(latencies.t_wait),
        t_pad=float(latencies.t_pad),
        t_lead=float(latencies.t_lead),
        t_present=float(latencies.t_present),
        t_total=float(latencies.t_total),
        no_fragment_at_admit=latencies.no_fragment_at_admit,
        clock_started=latencies.clock_started,
        consistent=latencies.consistent,
        arrival_monotonic=latencies.arrival_monotonic,
    )

# fmt: off
_METRIC_RENDER_SLEEP_FOR = _METER.create_histogram(
    name="rendering.loop.sleep_for",
    unit="s",
    description="Idle time between rendering loop iterations (bounded by present duration, ~200 ms)",
    explicit_bucket_boundaries_advisory=[0.001, 0.010, 0.030, 0.060, 0.080, 0.090, 0.095, 0.100, 0.105, 0.110, 0.120, 0.140, 0.170, 0.200],
)
# fmt: on


def _render_metric_attributes() -> dict[str, str]:
    attributes: dict[str, str] = {}
    stream_id = baggage.get_baggage("stream_id")
    if stream_id is not None:
        attributes["stream_id"] = str(stream_id)
    return attributes


def _buf2bytes(buffer: SpeechBuffer) -> bytes:
    return buffer.resample(constant.RENDERER_SPEECH_SAMPLE_RATE).to_bytes()


def _convert_config(config: RendererConfig) -> RenderConfig:
    extra_params = get_model_default_extra_params(config.model)
    extra_params.update(config.extra_params)

    return RenderConfig(
        avatar_id=config.avatar_id,
        background_id=config.background_id,
        pixel_format=config.pixel_format,
        height=constant.VIDEO_RESOLUTION[0],
        width=constant.VIDEO_RESOLUTION[1],
        extra_params=extra_params,
    )


@define
class RenderingWorklet:
    _renderer_registry: RendererClientRegistry
    _config: RendererConfig
    _segment_metadata: dict[SegmentId, SegmentMetadata] = Factory(dict)
    # Timing override, e.g. for a mock renderer with another N, T_future or T_lead.
    # None: the timing of the configured model.
    _durations: RenderingModelDurations | None = None

    def _get_durations(self) -> RenderingModelDurations:
        return self._durations or get_model_durations(self._config.model)

    @async_log_entry_exit
    async def run(self, bus: EventBus, clocks: StreamClocks) -> None:
        """
        Create schedulers, launch rendering loop as background task,
        then handle events in main loop.
        """
        durations = self._get_durations()
        speech_scheduler = SpeechScheduler(
            sample_rate=constant.NATIVE_SPEECH_SAMPLE_RATE,
            present_duration=durations.present,
            future_duration=durations.future,
            clocks=clocks,
        )
        user_speech_scheduler = SpeechScheduler(
            sample_rate=constant.RENDERER_SPEECH_SAMPLE_RATE,
            present_duration=durations.present,
            future_duration=durations.future,
            clocks=clocks,
        )

        async with TaskGroup() as tg:
            # Render loop is a publisher-only worker, needs its own clone
            tg.create_task(
                self._render_frames_loop(bus, clocks, speech_scheduler, user_speech_scheduler),
                name="RenderingWorklet._render_frames_loop",
            )

            await self._listen_bus(bus, speech_scheduler, user_speech_scheduler)

    async def _listen_bus(
        self,
        bus: EventBus,
        speech_scheduler: SpeechScheduler,
        user_speech_scheduler: SpeechScheduler,
    ):
        async with bus.subscribe(
            SegmentGenerationStarted,
            SegmentChunkGenerated,
            SegmentGenerationCompleted,
            UserSpeechReceived,
            UserSpeechStreamStart,
            UserSpeechStreamEnd,
            Shutdown,
            DiscardAvatarSpeechBuffer,
        ) as sub:
            bus.ready()  # Signal ready after subscribing
            async for event in sub:
                match event:
                    case SegmentGenerationStarted(segment_id=sid, metadata=meta):
                        # Store metadata for later use in playback events
                        self._segment_metadata[sid] = meta
                        await speech_scheduler.start_segment(sid)

                    case SegmentChunkGenerated(segment_id=sid, buffer=buf):
                        await speech_scheduler.append(buf, segment_id=sid)

                    case SegmentGenerationCompleted(segment_id=sid):
                        await speech_scheduler.end_segment(sid)

                    case UserSpeechReceived(buffer=buf):
                        await user_speech_scheduler.append(
                            buf.resample(constant.RENDERER_SPEECH_SAMPLE_RATE),
                            segment_id=SegmentId("user_speech"),
                        )

                    case UserSpeechStreamStart():
                        await user_speech_scheduler.start_segment(SegmentId("user_speech"))

                    case UserSpeechStreamEnd():
                        await user_speech_scheduler.end_segment(SegmentId("user_speech"))

                    case DiscardAvatarSpeechBuffer():
                        await speech_scheduler.interrupt()

                    case Shutdown():
                        await speech_scheduler.interrupt()
                        await speech_scheduler.stop()
                        await user_speech_scheduler.stop()
                        break

    def _get_and_remove_metadata(self, segment_id: SegmentId) -> SegmentMetadata:
        """Get metadata for segment and remove from cache."""
        return self._segment_metadata.pop(segment_id, {})

    async def _render_frames_loop(
        self,
        bus: EventBus,
        clocks: StreamClocks,
        speech_scheduler: SpeechScheduler,
        user_speech_scheduler: SpeechScheduler,
    ) -> None:
        """Main rendering loop."""
        state = None
        next_ts = Fraction()
        durations = self._get_durations()
        internal_config = _convert_config(self._config)
        renderer = self._renderer_registry.get_renderer(self._config.model)

        while not speech_scheduler.is_stopped():
            step_result = await speech_scheduler.do_step()
            user_chunk = await user_speech_scheduler.do_step()

            usd = float(user_speech_scheduler.unconsumed_duration)
            asd = float(speech_scheduler.unconsumed_duration)

            # Emit playback events from scheduler as delayed events
            for scheduler_event in step_result.events:
                match scheduler_event:
                    case TimestampedEvent(
                        event=SchedulerSegmentStarted(id=sid), timestamp=ts, latencies=lat
                    ):
                        # Get metadata but don't remove yet (will be removed on completion)
                        metadata = self._segment_metadata.get(SegmentId(sid), {})
                        await bus.publish(
                            ScheduledEvent(
                                emit_at=ts,
                                event=SegmentPlaybackStarted(
                                    segment_id=SegmentId(sid),
                                    metadata=metadata,
                                    latencies=_to_wire_latencies(lat),
                                ),
                            )
                        )
                    case TimestampedEvent(event=SchedulerSegmentEnded(id=sid), timestamp=ts):
                        # Remove metadata after segment completes
                        metadata = self._get_and_remove_metadata(SegmentId(sid))
                        await bus.publish(
                            ScheduledEvent(
                                emit_at=ts,
                                event=SegmentPlaybackCompleted(
                                    segment_id=SegmentId(sid), metadata=metadata
                                ),
                            )
                        )
                    case TimestampedEvent(
                        event=SegmentInterrupted(id=sid, played_duration=duration),
                        timestamp=ts,
                        latencies=lat,
                    ):
                        # Remove metadata after segment is interrupted
                        metadata = self._get_and_remove_metadata(SegmentId(sid))
                        await bus.publish(
                            ScheduledEvent(
                                emit_at=ts,
                                event=SegmentPlaybackInterrupted(
                                    segment_id=SegmentId(sid),
                                    played_duration=float(duration),
                                    metadata=metadata,
                                    latencies=_to_wire_latencies(lat),
                                ),
                            )
                        )
                    case TimestampedEvent(event=SegmentCancelled(segment_id=sid), timestamp=ts):
                        # Remove metadata after segment is cancelled
                        metadata = self._get_and_remove_metadata(SegmentId(sid))
                        await bus.publish(
                            ScheduledEvent(
                                emit_at=ts,
                                event=SegmentPlaybackCancelled(
                                    segment_id=SegmentId(sid), metadata=metadata
                                ),
                            )
                        )
                    case _:
                        _LOGGER.warning("Unexpected scheduler event: %r", scheduler_event)

            state, next_ts = await self._generate_frames(
                bus, renderer, state, next_ts, internal_config, step_result, user_chunk
            )

            start_next_render_at = float(max(next_ts - durations.lead, clocks.now))
            sleep_for = start_next_render_at - clocks.now

            _METRIC_RENDER_SLEEP_FOR.record(sleep_for, _render_metric_attributes())

            _LOGGER.debug(
                "Video worker iteration done",
                **{
                    "sleep_for": round(sleep_for, ndigits=4),
                    "next_ts": float(next_ts),
                    "now": round(clocks.now, ndigits=4),
                    "present_duration": float(step_result.present.duration),
                    "future_duration": float(step_result.future.duration),
                    "avatar_speech_duration_at_start": asd,
                    "user_speech_duration_at_start": usd,
                },
            )

            await clocks.wakeup_at(start_next_render_at)

    async def _generate_frames(
        self,
        bus: EventBus,
        renderer: AbstractRendererClient,
        renderer_state: bytes | None,
        first_frame_ts: Fraction,
        config: RenderConfig,
        audio: SpeechScheduler._StepResult,
        user_speech: SpeechScheduler._StepResult,
    ) -> tuple[bytes | None, Fraction]:
        """Generate frames and publish VideoFrameGenerated events."""
        async with renderer.generate(
            RendererRequest(
                current_chunk=await run_in_thread(_buf2bytes, audio.present),
                future_chunk=await run_in_thread(_buf2bytes, audio.future),
                current_chunk_listen=user_speech.present.to_bytes(),
                future_chunk_listen=user_speech.future.to_bytes(),
                timestamp_global=float(first_frame_ts),
                config=config,
                state=renderer_state,
            )
        ) as rsp:
            async for i, frame_bytes in aenumerate(rsp.frame_generator):
                past_frames_duration = i * constant.FRAME_DURATION
                frame_audio = audio.present.slice(
                    past_frames_duration, past_frames_duration + constant.FRAME_DURATION
                )

                if frame_audio.duration < constant.FRAME_DURATION:
                    _LOGGER.warning(
                        "Short buffer: last chunk duration %.3f which is less than a frame duration",
                        float(frame_audio.duration),
                    )

                frame = Frame(
                    buffer=frame_bytes,
                    timestamp=float(first_frame_ts + past_frames_duration),
                    audio=frame_audio,
                    pixel_format=config.pixel_format,
                )

                await bus.publish(VideoFrameGenerated(frame=frame))

            next_state = await rsp.state
            next_ts = first_frame_ts + constant.FRAME_DURATION * rsp.num_frames

            return next_state, next_ts
