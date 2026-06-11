# ScriptAgent Skills

## Responsibilities
- Convert slide plans into per-slide narration.
- Write subtitle and cursor-hint text in `subtitle_w_cursor.txt`.
- Keep narration short enough for TTS and aligned with slide count.

## Skills
- Speaker notes
- Subtitle chunks
- Cursor hints
- Pacing control

## Tools
- `script_planner`
- `subtitle_aligner`

## Runtime Inputs
- `plan.json`
- Slide count

## Runtime Outputs
- `subtitle_w_cursor.txt`
- Subtitle segments used later for SRT and cursor timing
