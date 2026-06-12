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

## Agentic Policy
- Run as a parallel branch from `PlannerAgent` beside `SlideBuilderAgent`.
- Produce narration state that can be consumed by both `SpeechAgent` and `GroundingAgent`.
- Shorten or split narration when target duration, subtitle readability, or TTS batch size is violated.
- Write narration as natural academic speech, not repeated labels or checklist markers.
- Avoid repeated evidence-label, ordinal-label, or takeaway-label openers.
- When a slide needs extra pacing words, add one contextual bridge sentence tied to the slide role.
- Route back to `PlannerAgent` if the plan lacks enough evidence for a faithful script.
- Publish cursor hints with every narration segment; missing hints are graph errors, not optional text.
