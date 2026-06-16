from __future__ import annotations

from src.aimooc_schema import AIMOOCSpec, AvatarConfig


def build_avatar_config(spec: AIMOOCSpec) -> AvatarConfig:
    mode = "none"
    if spec.include_avatar:
        mode = spec.avatar_mode or "presenter_card"
    return AvatarConfig(
        avatar_mode=mode,
        style=spec.preferred_style,
        expression_policy="calm" if spec.difficulty != "beginner" else "encouraging",
        gesture_policy="minimal",
        lip_sync=False,
    )
