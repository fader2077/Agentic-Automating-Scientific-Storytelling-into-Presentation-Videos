import os
import re
import subprocess
import torch
from os import path
from pathlib import Path


def load_whisperx():
    try:
        import whisperx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "whisperx is required only when ref_text is missing. "
            "Install whisperx or provide --ref_text."
        ) from exc
    return whisperx


def load_f5tts_class():
    try:
        from f5_tts.api import F5TTS
    except ModuleNotFoundError as exc:
        raise RuntimeError("f5_tts is required for model_type='f5'. Install F5TTS in the pipeline Python environment.") from exc
    return F5TTS


def transcribe_with_whisperx(audio_path, lang="en", device="cuda" if torch.cuda.is_available() else "cpu"):
    whisperx = load_whisperx()
    print(f"Using device: {device}")
    model = whisperx.load_model("large-v2", device=device, compute_type="float16" if device == "cuda" else "int8")
    result = model.transcribe(audio_path, language=lang)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device)
    segments = result_aligned["segments"]
    text = " ".join(seg["text"].strip() for seg in segments)
    return text

def inference_f5(text_prompt, save_path, ref_audio, ref_text, f5tts=None, speed=1.0):
    F5TTS = load_f5tts_class()
    synthesizer = f5tts or F5TTS()
    synthesizer.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=text_prompt,
        speed=speed,
        file_wave=save_path,
        seed=None,
    )

def parse_script(script_text):
    pages = script_text.strip().split("###\n")
    result = []
    for page in pages:
        if not page.strip(): continue
        lines = page.strip().split("\n")
        page_data = []
        for line in lines:
            if "|" not in line: 
                continue
            text, cursor = line.split("|", 1)
            page_data.append([text.strip(), cursor.strip()])
        result.append(page_data)
    return result


def split_narration_chunks(text, max_chars=260):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
    chunks = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks or [text]


def make_silence_wav(save_path, seconds, sample_rate=24000):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl=mono",
            "-t",
            f"{seconds:.3f}",
            "-acodec",
            "pcm_s16le",
            str(save_path),
        ],
        check=True,
        timeout=60,
    )


def concat_wavs(input_files, save_path):
    list_path = Path(save_path).with_suffix(".concat.txt")
    with open(list_path, "w", encoding="utf-8") as handle:
        for wav_path in input_files:
            normalized = str(Path(wav_path).resolve()).replace("\\", "/").replace("'", "'\\''")
            handle.write(f"file '{normalized}'\n")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-acodec",
            "pcm_s16le",
            str(save_path),
        ],
        check=True,
        timeout=180,
    )
    list_path.unlink(missing_ok=True)


def synthesize_slide_audio(
    model_type,
    script_path,
    speech_save_dir,
    ref_audio,
    ref_text=None,
    voice_speed=1.0,
    sentence_pause=0.0,
):
    with open(script_path, 'r') as f: script_with_cursor = ''.join(f.readlines())
    parsed_speech = parse_script(script_with_cursor)
    
    os.makedirs(speech_save_dir, exist_ok=True)
    if ref_text is None:
        ref_text = transcribe_with_whisperx(ref_audio)

    F5TTS = load_f5tts_class() if model_type == "f5" else None
    f5tts = F5TTS() if model_type == "f5" else None
    
    for slide_idx in range(len(parsed_speech)):
        speech_with_cursor = parsed_speech[slide_idx]
        subtitle = ""
        for sentence_idx, (prompt, cursor_prompt) in enumerate(speech_with_cursor):
            if len(subtitle) == 0: subtitle = prompt
            else: subtitle = subtitle + "\n\n\n" + prompt
        speech_result_path = path.join(speech_save_dir, "{}.wav".format(str(slide_idx)))
        if model_type == "f5":
            chunks = split_narration_chunks(subtitle)
            if len(chunks) == 1 and sentence_pause <= 0:
                inference_f5(chunks[0], speech_result_path, ref_audio, ref_text, f5tts=f5tts, speed=voice_speed)
                continue

            chunk_dir = Path(speech_save_dir) / "_chunks" / str(slide_idx)
            os.makedirs(chunk_dir, exist_ok=True)
            concat_inputs = []
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_path = chunk_dir / f"{chunk_idx:03d}.wav"
                inference_f5(chunk_text, str(chunk_path), ref_audio, ref_text, f5tts=f5tts, speed=voice_speed)
                concat_inputs.append(chunk_path)
                if sentence_pause > 0 and chunk_idx < len(chunks) - 1:
                    pause_path = chunk_dir / f"{chunk_idx:03d}_pause.wav"
                    make_silence_wav(pause_path, sentence_pause)
                    concat_inputs.append(pause_path)
            concat_wavs(concat_inputs, speech_result_path)


def tts_per_slide(model_type, script_path, speech_save_dir, ref_audio, ref_text=None):
    return synthesize_slide_audio(model_type, script_path, speech_save_dir, ref_audio, ref_text=ref_text)
