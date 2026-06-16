# CoursePackagerAgent Skills

- Collect course plan, source manifest, course spec, lesson artifacts, feedback rounds, and avatar config.
- Write `course_package_manifest.json`.
- Include `course_video_manifest.json` when multi-source PDF video generation is requested.
- Ensure generated lesson media is listed in package artifacts, including `video.mp4` and avatar-integrated `avatar_video.mp4`.
- Keep package layout stable for export and future LMS integration.
- Verify paths are package-local before serving artifacts.
