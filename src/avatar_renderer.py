from __future__ import annotations

import subprocess
from pathlib import Path

try:
    from src.aimooc_schema import AvatarConfig, write_json
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from aimooc_schema import AvatarConfig, write_json


ROOT = Path(__file__).resolve().parents[1]
AVATAR_DIR = ROOT / "web" / "avatar"


def discover_avatar_image(explicit_path: str | Path | None = None) -> Path | None:
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists() and candidate.is_file():
            return candidate
    preferred = AVATAR_DIR / "kafka.jpg"
    if preferred.exists():
        return preferred
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        matches = sorted(AVATAR_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def compose_avatar_overlay(
    source_video: str | Path,
    output_video: str | Path,
    avatar_image: str | Path | None = None,
    position: str = "bottom_right",
) -> dict[str, object]:
    source = Path(source_video)
    output = Path(output_video)
    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    image = discover_avatar_image(avatar_image)
    if position not in {"bottom_right", "bottom_left"}:
        position = "bottom_right"

    card_x = "iw-260" if position == "bottom_right" else "35"
    avatar_x = "main_w-235" if position == "bottom_right" else "60"
    fallback_avatar_x = "iw-235" if position == "bottom_right" else "60"
    filter_chain = (
        f"drawbox=x={card_x}:y=ih-235:w=225:h=200:color=0x0f172a@0.72:t=fill,"
        f"drawbox=x={card_x}:y=ih-235:w=225:h=200:color=0x38bdf8@0.45:t=2"
    )
    if image:
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-loop",
            "1",
            "-i",
            str(image),
            "-filter_complex",
            (
                "[1:v]scale=165:165:force_original_aspect_ratio=decrease,"
                "pad=165:165:(ow-iw)/2:(oh-ih)/2:color=black@0.0,format=rgba[avatar];"
                f"[0:v]{filter_chain}[base];"
                f"[base][avatar]overlay=x={avatar_x}:y=main_h-215:shortest=1[v]"
            ),
            "-map",
            "[v]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "copy",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(output),
        ]
    else:
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            filter_chain + f",drawbox=x={fallback_avatar_x}:y=ih-215:w=165:h=165:color=0xe2e8f0@0.95:t=fill",
            "-map",
            "0:v",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "copy",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(output),
        ]
    subprocess.run(command, check=True, timeout=600)
    return {
        "source_video": str(source),
        "video": str(output),
        "avatar_image": str(image) if image else "",
        "position": position,
    }


def render_presenter_card_video(lesson_dir: Path, config: AvatarConfig, duration: int = 3) -> Path:
    video_path = lesson_dir / "avatar_video.mp4"
    lesson_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x111827:s=1280x720:d={duration}",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-vf",
            "drawbox=x=930:y=430:w=260:h=220:color=0x2563eb@0.92:t=fill,"
            "drawbox=x=960:y=460:w=80:h=80:color=white@0.95:t=fill",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        check=True,
        timeout=60,
    )
    return video_path


def render_avatar_manifest(
    lesson_dir: Path,
    config: AvatarConfig,
    render_media: bool = False,
    source_video: str | Path | None = None,
    avatar_image: str | Path | None = None,
) -> dict[str, object]:
    output = {
        "avatar_mode": config.avatar_mode,
        "avatar_id": config.avatar_id,
        "position": config.position,
        "rendered": False,
        "video": "",
        "source_video": str(source_video) if source_video else "",
        "avatar_image": "",
    }
    if config.avatar_mode == "none":
        write_json(lesson_dir / "avatar_manifest.json", output)
        return output
    if render_media and config.avatar_mode == "presenter_card":
        try:
            if source_video:
                result = compose_avatar_overlay(
                    source_video,
                    lesson_dir / "avatar_video.mp4",
                    avatar_image=avatar_image,
                    position=config.position,
                )
                output.update(result)
            else:
                video_path = render_presenter_card_video(lesson_dir, config)
                output["video"] = str(video_path)
                image = discover_avatar_image(avatar_image)
                output["avatar_image"] = str(image) if image else ""
            output["rendered"] = True
        except Exception as exc:
            output["error"] = str(exc)
    write_json(lesson_dir / "avatar_manifest.json", output)
    return output
