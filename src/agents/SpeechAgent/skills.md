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
