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
