from __future__ import annotations

import json
import subprocess
from pathlib import Path


def run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def get_video_resolution(path: str) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    return int(stream["width"]), int(stream["height"])


def get_video_duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return max(float(result.stdout.strip() or 0), 0.0)


def copy_video(input_video: str, output_video: str) -> None:
    run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error", "-i", input_video, "-c", "copy", output_video])


def drawbox_filter(x_expr: str, y_expr: str, start: float, end: float, cursor_size: int) -> str:
    start = round(max(start, 0.0), 3)
    end = round(max(end, start), 3)
    return (
        f"drawbox=x={x_expr}:y={y_expr}:w={cursor_size}:h={cursor_size}:"
        f"color=red@0.95:t=fill:enable='between(t,{start},{end})'"
    )


def render_cursor_on_video(
    input_video: str,
    output_video: str,
    cursor_points: list[tuple[float, float, float]],
    transition_duration: float = 0.1,
    cursor_size: int = 10,
    cursor_img_path: str | None = None,
) -> None:
    del cursor_img_path
    if not cursor_points:
        copy_video(input_video, output_video)
        return

    width, height = get_video_resolution(input_video)
    duration = get_video_duration(input_video)
    size = max(int(cursor_size), 2)
    half = size / 2
    max_x = max(width - size, 0)
    max_y = max(height - size, 0)

    def point_expr(x: float, y: float) -> tuple[str, str]:
        return str(round(min(max(x - half, 0), max_x), 3)), str(round(min(max(y - half, 0), max_y), 3))

    filters: list[str] = []
    points = sorted((float(t), float(x), float(y)) for t, x, y in cursor_points)
    transition = max(float(transition_duration), 0.001)

    first_time, first_x, first_y = points[0]
    first_expr = point_expr(first_x, first_y)
    if first_time > 0:
        filters.append(drawbox_filter(first_expr[0], first_expr[1], 0.0, first_time, size))

    for idx, (current_time, current_x, current_y) in enumerate(points):
        next_point = points[idx + 1] if idx + 1 < len(points) else None
        current_expr = point_expr(current_x, current_y)
        if not next_point:
            filters.append(drawbox_filter(current_expr[0], current_expr[1], current_time, duration, size))
            continue

        next_time, next_x, next_y = next_point
        move_start = max(next_time - transition, current_time)
        if move_start > current_time:
            filters.append(drawbox_filter(current_expr[0], current_expr[1], current_time, move_start, size))

        move_len = max(next_time - move_start, 0.001)
        start_x = min(max(current_x - half, 0), max_x)
        start_y = min(max(current_y - half, 0), max_y)
        end_x = min(max(next_x - half, 0), max_x)
        end_y = min(max(next_y - half, 0), max_y)
        x_expr = f"{round(start_x, 3)}+({round(end_x - start_x, 3)})*(t-{round(move_start, 3)})/{round(move_len, 3)}"
        y_expr = f"{round(start_y, 3)}+({round(end_y - start_y, 3)})*(t-{round(move_start, 3)})/{round(move_len, 3)}"
        filters.append(drawbox_filter(x_expr, y_expr, move_start, next_time, size))

    filter_chain = ",".join(filters)
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            input_video,
            "-vf",
            filter_chain,
            "-map",
            "0:v",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            output_video,
        ]
    )


def render_video_with_cursor_from_json(
    video_path: str,
    out_video_path: str,
    json_path: str,
    cursor_img_path: str | None = None,
    transition_duration: float = 0.1,
    cursor_size: int = 16,
) -> None:
    del cursor_img_path
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    cursor_points: list[tuple[float, float, float]] = []
    for idx, slide in enumerate(data):
        start_time = float(slide["start"]) if idx == 0 else float(slide["start"]) + 0.5
        x, y = slide["cursor"]
        cursor_points.append((start_time, float(x), float(y)))

    render_cursor_on_video(
        input_video=video_path,
        output_video=out_video_path,
        cursor_points=cursor_points,
        transition_duration=transition_duration,
        cursor_size=cursor_size,
    )
