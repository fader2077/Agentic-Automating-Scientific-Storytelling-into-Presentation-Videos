# LessonBuilderAgent Skills

- Build each lesson with objectives, slides, narration script, quiz, and manifest.
- Keep lesson artifacts local to the lesson directory.
- When media rendering is requested, call the course video pipeline on the multi-source PDF bundle instead of only reusing an existing single-paper video.
- Attach generated `video.mp4`, `avatar_video.mp4`, and `avatar_manifest.json` to the lesson manifest.
- Avoid overwriting previous version directories.
