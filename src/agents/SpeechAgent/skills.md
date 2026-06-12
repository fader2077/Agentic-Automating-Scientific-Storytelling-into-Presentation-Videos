# SpeechAgent Skills

## Responsibilities
- Load the configured F5TTS engine.
- Apply the reference audio and reference text.
- Synthesize one audio file per slide.
- Use CUDA when the local PyTorch runtime exposes a GPU.

## Skills
- Voice conditioning
- F5TTS synthesis
- Per-slide audio packaging

## Tools
- `voice_profiler`
- `f5_queue`

## Runtime Inputs
- `subtitle_w_cursor.txt`
- Reference WAV path
- Reference transcription text

## Runtime Outputs
- `audio/*.wav`

## Agentic Policy
- Enter after `ScriptAgent` publishes narration segments.
- Prefer user-provided reference audio and text; use packaged F5TTS reference voice only when no configured reference exists.
- Normalize total audio duration against the requested target minutes before handing off.
- Report TTS dependency, CUDA, or duration errors as graph state for repair instead of hiding them.
- Publish `audio` only after every slide has a matching WAV file.
