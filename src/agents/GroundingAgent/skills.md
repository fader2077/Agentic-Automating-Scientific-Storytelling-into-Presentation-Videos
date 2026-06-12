# GroundingAgent Skills

## Responsibilities
- Read slide images and generated audio durations.
- Convert cursor hints into slide coordinates.
- Write cursor timeline JSON for the video overlay.

## Skills
- Slide region reading
- Cursor trajectory
- Timing alignment

## Tools
- `layout_inspector`
- `cursor_router`

## Runtime Inputs
- `subtitle_w_cursor.txt`
- `slide_imgs/*.png`
- `audio/*.wav`

## Runtime Outputs
- `cursor.json`
- `cursor_mid.json`

## Agentic Policy
- Enter through `ArtifactJoinAgent` only after slide images and audio are both present.
- Use slide dimensions and narration cursor hints to compute deterministic focus paths.
- Reject mismatched slide/audio counts instead of guessing timings.
- Publish `cursor` state only when every slide has a cursor beat.
- Route failures to render blocking state, because cursor errors affect final MP4 composition.
