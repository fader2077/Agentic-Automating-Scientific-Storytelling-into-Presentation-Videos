import os
import torch
import whisperx
from os import path
from f5_tts.api import F5TTS


def transcribe_with_whisperx(audio_path, lang="en", device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Using device: {device}")
    model = whisperx.load_model("large-v2", device=device, compute_type="float16" if device == "cuda" else "int8")
    result = model.transcribe(audio_path, language=lang)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device)
    segments = result_aligned["segments"]
    text = " ".join(seg["text"].strip() for seg in segments)
    return text

def inference_f5(text_prompt, save_path, ref_audio, ref_text, f5tts=None):
    synthesizer = f5tts or F5TTS()
    synthesizer.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=text_prompt,
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

def tts_per_slide(model_type, script_path, speech_save_dir, ref_audio, ref_text=None):
    with open(script_path, 'r') as f: script_with_cursor = ''.join(f.readlines())
    parsed_speech = parse_script(script_with_cursor)
    
    os.makedirs(speech_save_dir, exist_ok=True)
    if ref_text is None:
        ref_text = transcribe_with_whisperx(ref_audio)

    f5tts = F5TTS() if model_type == "f5" else None
    
    for slide_idx in range(len(parsed_speech)):
        speech_with_cursor = parsed_speech[slide_idx]
        subtitle = ""
        for sentence_idx, (prompt, cursor_prompt) in enumerate(speech_with_cursor):
            if len(subtitle) == 0: subtitle = prompt
            else: subtitle = subtitle + "\n\n\n" + prompt
        speech_result_path = path.join(speech_save_dir, "{}.wav".format(str(slide_idx)))
        if model_type == "f5":
            inference_f5(subtitle, speech_result_path, ref_audio, ref_text, f5tts=f5tts)
