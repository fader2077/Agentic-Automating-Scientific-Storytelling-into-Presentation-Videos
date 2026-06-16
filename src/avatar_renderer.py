from __future__ import annotations

import subprocess
from pathlib import Path

from src.aimooc_schema import AvatarConfig, write_json


def render_avatar_manifest(lesson_dir: Path, config: AvatarConfig, render_media: bool = False) -> dict[str, object]:
    output = {
        "avatar_mode": config.avatar_mode,
        "avatar_id": config.avatar_id,
        "position": config.position,
        "rendered": False,
        "video": "",
    }
    if render_media and config.avatar_mode == "presenter_card":
        video_path = lesson_dir / "avatar_video.mp4"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=0x111827:s=1280x720:d=3",
                    "-vf",
                    "drawbox=x=930:y=430:w=260:h=220:color=0x2563eb@0.92:t=fill,drawbox=x=960:y=460:w=80:h=80:color=white@0.95:t=fill",
                    "-pix_fmt",
                    "yuv420p",
                    str(video_path),
                ],
                check=True,
                timeout=30,
            )
            output["rendered"] = True
            output["video"] = str(video_path)
        except Exception as exc:
            output["error"] = str(exc)
    write_json(lesson_dir / "avatar_manifest.json", output)
    return output
