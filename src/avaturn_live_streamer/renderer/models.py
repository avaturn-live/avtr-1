# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from fractions import Fraction
from typing import Literal

import attrs

from avaturn_live_streamer.constant import FRAME_DURATION, RENDERER_SPEECH_SAMPLE_RATE
from avaturn_live_streamer.renderer.interface import RendererParamValue

ModelName = Literal["avtrn-1", "delta-legacy"]
PublicModelName = Literal["delta", "golf"]


def public_model_name_to_model_name(name: PublicModelName) -> ModelName:
    # Both "delta" and "golf" map to "avtrn-1" for backwards compatibility
    return "avtrn-1"


@attrs.define
class RenderingModelDurations:
    """Timing of one renderer, as the streamer uses it.

    present: T_present, the audio one request turns into frames (N frames of FRAME_DURATION).
    future: T_future, the lookahead audio a request needs after its present window.
    lead: T_lead, the rendering loop issues each request this long before its present window
        plays. It must cover the renderer's time to first frame.
    """

    present: Fraction
    future: Fraction
    lead: Fraction


_DURATION_BY_MODEL: dict[ModelName, RenderingModelDurations] = {
    "avtrn-1": RenderingModelDurations(
        present=FRAME_DURATION * 5,
        # 2 frames + 80 samples in 16kHz
        future=FRAME_DURATION * 5 + Fraction(80, RENDERER_SPEECH_SAMPLE_RATE),
        lead=Fraction(100, 1000),
    ),
}


def get_model_durations(model: ModelName) -> RenderingModelDurations:
    return _DURATION_BY_MODEL[model]


def get_model_default_extra_params(model: ModelName) -> dict[str, RendererParamValue]:
    if model == "avtrn-1":
        return {
            "cfg_self_audio": 2.0,
            "cfg_other_audio": 2.0,
            "cfg_kp": 3.0,
            "noise_alpha": 2.0,
            "noise_trunc_z": 1.2,
            "play_intro": False,
        }
    return {}
