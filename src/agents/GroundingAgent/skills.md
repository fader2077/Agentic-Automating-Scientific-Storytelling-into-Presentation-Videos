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
