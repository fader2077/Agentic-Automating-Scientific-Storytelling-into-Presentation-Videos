# RenderAgent Skills

## Responsibilities
- Build per-slide MP4 clips from slide images and audio.
- Merge clips into a single video.
- Overlay the cursor track.
- Burn subtitles into the final MP4 and verify media duration.

## Skills
- FFmpeg composition
- Subtitle burn-in
- MP4 verification
- Artifact handoff

## Tools
- `ffmpeg_packager`
- `mp4_verifier`

## Runtime Inputs
- `slide_imgs/*.png`
- `audio/*.wav`
- `cursor.json`
- `subtitles.srt`

## Runtime Outputs
- `1_merage.mp4`
- `2_merage.mp4`
- `3_merage.mp4`
- `sat.json`
- `token.json`

## Agentic Policy
- Enter only after `GroundingAgent` publishes cursor state.
- Verify required assets exist, including `src/cursor_image/red.png`, before composition.
- Package slides, audio, cursor overlay, and subtitles into the final MP4.
- Verify final duration with `ffprobe` and report mismatch against target minutes.
- Publish final artifacts only after MP4, subtitles, cursor JSON, SAT, and token metadata exist.
