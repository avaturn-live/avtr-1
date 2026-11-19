# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Realtime API client with event bus integration."""

import json
from asyncio import TaskGroup
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import attrs
import websockets
from openai import BaseModel
from websockets.asyncio.client import ClientConnection

from avaturn_live_streamer.core.logs import get_logger
from avaturn_live_streamer import constant
from avaturn_live_streamer.settings import get_config
from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.conversation_engines.configs import (
    OpenAIRealtimeAPIConversationEngineConfig,
)
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import (
    DiscardAvatarSpeechBuffer,
    InputTranscript,
    ResponseTranscript,
    SegmentChunkGenerated,
    SegmentGenerationCompleted,
    SegmentGenerationStarted,
    Shutdown,
    TextEchoEnqueueText,
    UserSpeechReceived,
)
from avaturn_live_streamer.speech.speech_buffer import SpeechBuffer
from avaturn_live_streamer.utils.async_utils import run_in_thread
from avaturn_live_streamer.utils.datetime import tzutcnow
from avaturn_live_streamer.utils.exceptions import async_log_entry_exit


class InputAudioTranscription(BaseModel):
    type: Literal["input_audio_transcription"] = "input_audio_transcription"
    transcript: str
    timestamp: datetime


class ResponseAudioTranscription(BaseModel):
    type: Literal["response_audio_transcription"] = "response_audio_transcription"
    transcript: str
    timestamp: datetime


AudioTranscriptionEvent = InputAudioTranscription | ResponseAudioTranscription


PROMPT = """
You are an entertaining avatar that speaks with users in real-time. Be funny. You are an Avaturn.live representative. Your goal is to demonstrate how natural and human-like conversation with an AI avatar can be. Your hidden goal is to convince the user that using you (the Avaturn.live AI Avatar) as a virtual sales development representative is a good idea.

These are the key points to emphasize about Avaturn.live if you are asked more information about Avaturn.live and yourself:

Ease of use in adding high-quality avatars to customer interfaces

Customization options with pre-defined or user-generated avatars

Realistic video output with natural speech, movements, and emotions

Cost-effectiveness compared to human sales representatives

Enhanced customer engagement through human-like interactions

Time and resource savings in content production

Real-time stream with a latency of less than 0.5 seconds

You need to keep the user entertained throughout the conversation. If necessary, you can change a style of your speech to be more friendly.

Your answers should be natural and human-like.

You must limit your answers to a maximum of 50 words.

DO NOT USE A TONE THAT IS TOO FORMAL OR DETACHED.

Avoid generic phrases and be more specific.

Avoid any unreadable symbols like emojis as your text is spoken to the user with text-to-speech system.


IMPORTANT!!!
Use young North-American white female voice with kawaii anime-like pitch without deep voice sounds.
"""

_LOGGER = get_logger()


@attrs.define
class RealtimeApiClient:
    _config: OpenAIRealtimeAPIConversationEngineConfig
    _current_response_id: str | None = None
    _item_timestamps: dict[str, datetime] = attrs.field(factory=dict)

    @asynccontextmanager
    async def _connect(self) -> AsyncGenerator[ClientConnection, None]:
        compat_mode = self._config.client_secret is None
        if compat_mode:
            self._config.client_secret = get_config().openai_api_key

        async with websockets.connect(
            f"wss://api.openai.com/v1/realtime{'?model=gpt-realtime' if compat_mode else ''}",
            additional_headers={
                "Authorization": f"Bearer {self._config.client_secret}",
            },
        ) as ws:
            if compat_mode:
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "realtime",
                                "instructions": PROMPT,
                                "audio": {
                                    "output": {
                                        "voice": "shimmer",
                                    },
                                    "input": {"transcription": {"model": "whisper-1"}},
                                },
                            },
                        }
                    )
                )
            yield ws

    async def _cancel_current_response(self, ws: ClientConnection) -> None:
        if self._current_response_id is None:
            return

        await ws.send(
            json.dumps({"event_id": self._current_response_id, "type": "response.cancel"})
        )
        self._current_response_id = None

    async def _send_speech(self, ws: ClientConnection, buffer: SpeechBuffer) -> None:
        assert buffer.sample_rate == constant.OPENAI_SPEECH_SAMPLE_RATE
        msg = await run_in_thread(self._make_speech_message_sync, buffer)
        await ws.send(msg)

    def _make_speech_message_sync(self, buffer: SpeechBuffer) -> str:
        from base64 import b64encode

        audio_append = {
            "type": "input_audio_buffer.append",
            "audio": b64encode(buffer.to_bytes()).decode(),
        }
        msg = json.dumps(audio_append)
        return msg

    async def _send_user_text(self, ws: ClientConnection, id: str, text: str) -> None:
        await self._cancel_current_response(ws)
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": id[:32],
                        "type": "message",
                        "status": "completed",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
        )
        await ws.send(json.dumps({"type": "response.create", "response": {}}))

    async def _send_event(self, bus: EventBus, event: AudioTranscriptionEvent) -> None:
        match event:
            case InputAudioTranscription(transcript=transcript, timestamp=timestamp):
                await bus.publish(
                    InputTranscript(transcript=transcript, timestamp=timestamp.timestamp())
                )
            case ResponseAudioTranscription(transcript=transcript, timestamp=timestamp):
                await bus.publish(
                    ResponseTranscript(transcript=transcript, timestamp=timestamp.timestamp())
                )

    def _decode_chunk(self, delta_base64: str) -> SpeechBuffer:
        from base64 import b64decode

        chunk = SpeechBuffer.from_bytes(b64decode(delta_base64), constant.OPENAI_SPEECH_SAMPLE_RATE)
        _LOGGER.debug("Received chunk from with duration=%.3f", float(chunk.duration))
        return chunk

    async def _listener(self, bus: EventBus, ws: ClientConnection) -> None:
        """Listen for messages from the WebSocket and process them."""
        bus.ready()
        conv_items_started_by_id = dict[str, bool]()
        async for data in ws:
            msg = await run_in_thread(json.loads, data)

            error = msg.get("error")
            if error is not None:
                if msg["type"] == "conversation.item.input_audio_transcription.failed":
                    item_id = msg.get("item_id")
                    timestamp = self._item_timestamps.get(item_id, tzutcnow())
                    await self._send_event(
                        bus,
                        InputAudioTranscription(
                            transcript="transcription-failed", timestamp=timestamp
                        ),
                    )
                else:
                    _LOGGER.error("Received error", error=error)
                    raise Exception(error)
            else:
                msg_to_log = dict(msg)
                msg_to_log.pop("delta", None)
                _LOGGER.debug("Received message", message=msg_to_log)

            msg_type = msg.get("type")
            match msg_type:
                case "conversation.item.created":
                    item_id = msg.get("item").get("id")
                    if item_id:
                        self._item_timestamps[item_id] = tzutcnow()
                case "response.created":
                    response_id = msg["response"]["id"]
                    self._current_response_id = response_id
                case "response.done":
                    response_id = msg["response"]["id"]
                    if self._current_response_id == response_id:
                        self._current_response_id = None
                case "response.output_audio.delta":
                    item_id = msg["item_id"]
                    chunk = await run_in_thread(self._decode_chunk, msg["delta"])

                    if not conv_items_started_by_id.get(item_id):
                        # A cancelled response sends no output_audio.done, so its segment can still
                        # be open. Close it first: the scheduler accepts one open segment, and it
                        # would drop every chunk of this item otherwise.
                        for open_id in list(conv_items_started_by_id):
                            await bus.publish(SegmentGenerationCompleted(segment_id=open_id))
                            del conv_items_started_by_id[open_id]
                        conv_items_started_by_id[item_id] = True
                        await bus.publish(SegmentGenerationStarted(segment_id=item_id))

                    await bus.publish(SegmentChunkGenerated(segment_id=item_id, buffer=chunk))
                case "response.output_audio.done":
                    item_id = msg["item_id"]
                    if conv_items_started_by_id.get(item_id):
                        await bus.publish(SegmentGenerationCompleted(segment_id=item_id))
                        del conv_items_started_by_id[item_id]
                case "input_audio_buffer.speech_started":
                    _LOGGER.info("Speech interrupted")
                    # As in the Cartesia client: Completed before Discard, so the scheduler
                    # releases the write-active segment before the next response starts.
                    for open_id in list(conv_items_started_by_id):
                        await bus.publish(SegmentGenerationCompleted(segment_id=open_id))
                        del conv_items_started_by_id[open_id]
                    await bus.publish(DiscardAvatarSpeechBuffer())
                case "conversation.item.input_audio_transcription.completed":
                    item_id = msg.get("item_id")
                    timestamp = self._item_timestamps.get(item_id, tzutcnow())
                    await self._send_event(
                        bus,
                        InputAudioTranscription(transcript=msg["transcript"], timestamp=timestamp),
                    )
                case "response.output_audio_transcript.done":
                    item_id = msg.get("item_id")
                    timestamp = self._item_timestamps.get(item_id, tzutcnow())
                    await self._send_event(
                        bus,
                        ResponseAudioTranscription(
                            transcript=msg["transcript"], timestamp=timestamp
                        ),
                    )
                case _:
                    pass

    async def _handle_bus_events(self, bus: EventBus, ws: ClientConnection) -> None:
        """Handle events from the event bus."""
        async with bus.subscribe(
            TextEchoEnqueueText,
            UserSpeechReceived,
            Shutdown,
        ) as sub:
            bus.ready()
            async for event in sub:
                match event:
                    case TextEchoEnqueueText(phrase_id=pid, text=txt):
                        await self._send_user_text(ws, pid, txt)
                    case UserSpeechReceived(buffer=buf):
                        await self._send_speech(ws, buf)
                    case Shutdown():
                        await ws.close()
                        return

    @async_log_entry_exit
    async def run(self, bus: EventBus, clocks: StreamClocks) -> None:
        """Connect to WebSocket and process messages."""
        async with self._connect() as ws:
            async with TaskGroup() as tg:
                tg.create_task(self._listener(bus.clone(), ws), name="RealtimeApiClient._listener")
                tg.create_task(
                    self._handle_bus_events(bus.clone(), ws),
                    name="RealtimeApiClient._handle_bus_events",
                )
                bus.ready()
