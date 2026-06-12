import re
import os
import json
import subprocess
from os import path


def parse_script(script_text):
    """
    Parse subtitle_w_cursor.txt.

    Expected format:

    sentence | cursor description
    sentence | cursor description
    ###
    sentence | cursor description
    """
    pages = script_text.strip().split("###")
    result = []

    for page in pages:
        if not page.strip():
            continue

        lines = page.strip().split("\n")
        page_data = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if "|" in line:
                text, cursor = line.split("|", 1)
                page_data.append([text.strip(), cursor.strip()])
            else:
                # Fallback if a line has no cursor prompt
                page_data.append([line.strip(), "center of slide"])

        if page_data:
            result.append(page_data)

    return result


def get_audio_length(audio_path):
    """
    Get audio duration in seconds using ffprobe first, fallback to ffmpeg.
    """
    if not path.exists(audio_path):
        return 3.0

    # Preferred: ffprobe
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        duration = float(result.stdout.strip())

        if duration > 0:
            return duration

    except Exception:
        pass

    # Fallback: ffmpeg stderr parsing
    try:
        cmd = ["ffmpeg", "-i", audio_path]
        result = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            text=True,
        )

        for line in result.stderr.splitlines():
            if "Duration" in line:
                duration_str = line.split("Duration:")[1].split(",")[0].strip()
                hours, minutes, seconds = duration_str.split(":")
                return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

    except Exception:
        pass

    return 3.0


def get_slide_size(slide_img_dir):
    """
    Get slide size from the first slide image.
    Return width, height.
    """
    try:
        import cv2

        imgs = [
            name for name in os.listdir(slide_img_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        imgs = sorted(
            imgs,
            key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 9999,
        )

        if not imgs:
            return 1600, 900

        img_path = path.join(slide_img_dir, imgs[0])
        img = cv2.imread(img_path)

        if img is None:
            return 1600, 900

        h, w = img.shape[:2]
        return w, h

    except Exception:
        return 1600, 900


def estimate_cursor_position(cursor_prompt, slide_w, slide_h):
    """
    Deterministic cursor position based on cursor prompt.
    Coordinates are in slide image pixel space.
    """
    prompt = (cursor_prompt or "").lower()

    center = (slide_w // 2, slide_h // 2)
    title = (slide_w // 2, int(slide_h * 0.18))
    bullet_left = (int(slide_w * 0.25), int(slide_h * 0.42))
    bullet_mid = (int(slide_w * 0.32), int(slide_h * 0.52))
    figure = (int(slide_w * 0.55), int(slide_h * 0.62))
    right_figure = (int(slide_w * 0.68), int(slide_h * 0.55))
    bottom = (slide_w // 2, int(slide_h * 0.78))

    if "title" in prompt:
        return title

    if (
        "figure" in prompt
        or "visual" in prompt
        or "diagram" in prompt
        or "image" in prompt
        or "chart" in prompt
        or "plot" in prompt
    ):
        return figure

    if "right" in prompt:
        return right_figure

    if (
        "bullet" in prompt
        or "list" in prompt
        or "main content" in prompt
        or "first bullet" in prompt
    ):
        return bullet_left

    if "center" in prompt:
        return center

    if "bottom" in prompt or "footer" in prompt:
        return bottom

    return bullet_mid


def sorted_audio_files(slide_audio_dir):
    """
    Sort generated slide audio files by leading number.
    """
    files = [
        name for name in os.listdir(slide_audio_dir)
        if name.lower().endswith((".wav", ".mp3", ".flac", ".m4a"))
    ]

    files = sorted(
        files,
        key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 9999,
    )

    return [path.join(slide_audio_dir, name) for name in files]


def build_cursor_timeline(
    script_path,
    slide_img_dir,
    slide_audio_dir,
    cursor_save_path,
    gpu_list=None,
):
    """
    Deterministic cursor generation.

    This version avoids:
    - multiprocessing
    - UI-TARS model loading
    - WhisperX alignment
    - torch CUDA child-process initialization
    - PYTHONHASHSEED crash

    It creates cursor timestamps by evenly dividing each slide audio duration
    across the sentences generated for that slide.
    """
    with open(script_path, "r", encoding="utf-8") as f:
        script_with_cursor = f.read()

    parsed_speech = parse_script(script_with_cursor)

    slide_w, slide_h = get_slide_size(slide_img_dir)
    audio_files = sorted_audio_files(slide_audio_dir)

    if len(audio_files) == 0:
        raise RuntimeError(f"No audio files found in {slide_audio_dir}")

    output = []
    mid_output = []

    global_time = 0.0
    slide_count = min(len(parsed_speech), len(audio_files))

    if slide_count == 0:
        raise RuntimeError(
            f"No matching slides/audio found. "
            f"parsed_speech={len(parsed_speech)}, audio_files={len(audio_files)}"
        )

    for slide_idx in range(slide_count):
        speech_with_cursor = parsed_speech[slide_idx]
        audio_path = audio_files[slide_idx]

        duration = get_audio_length(audio_path)

        if duration <= 0:
            duration = max(3.0, len(speech_with_cursor) * 2.0)

        n = max(1, len(speech_with_cursor))
        seg_len = duration / n

        for sent_idx, (speech_text, cursor_prompt) in enumerate(speech_with_cursor):
            start = global_time + sent_idx * seg_len
            end = global_time + (sent_idx + 1) * seg_len

            cursor = estimate_cursor_position(
                cursor_prompt=cursor_prompt,
                slide_w=slide_w,
                slide_h=slide_h,
            )

            item = {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": speech_text,
                "cursor": cursor,
            }

            output.append(item)

            mid_output.append({
                "slide": slide_idx,
                "sentence": sent_idx,
                "speech_text": speech_text,
                "cursor_prompt": cursor_prompt,
                "cursor": cursor,
                "token": "",
            })

        global_time += duration

    os.makedirs(path.dirname(cursor_save_path), exist_ok=True)

    with open(cursor_save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(cursor_save_path.replace(".json", "_mid.json"), "w", encoding="utf-8") as f:
        json.dump(mid_output, f, indent=2, ensure_ascii=False)

    print(f"Deterministic cursor JSON saved to: {cursor_save_path}")
    print(f"Deterministic cursor mid JSON saved to: {cursor_save_path.replace('.json', '_mid.json')}")
    print(f"Total cursor points: {len(output)}")

    # Original function returns token usage estimate.
    return 0


def cursor_gen_per_sentence(
    script_path,
    slide_img_dir,
    slide_audio_dir,
    cursor_save_path,
    gpu_list=None,
):
    return build_cursor_timeline(
        script_path=script_path,
        slide_img_dir=slide_img_dir,
        slide_audio_dir=slide_audio_dir,
        cursor_save_path=cursor_save_path,
        gpu_list=gpu_list,
    )
