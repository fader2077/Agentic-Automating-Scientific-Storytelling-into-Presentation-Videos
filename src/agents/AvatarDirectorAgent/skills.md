# AvatarDirectorAgent Skills

- Decide avatar mode from course spec: `none`, `presenter_card`, or optional `talking_head`.
- Keep avatar rendering independent from slide rendering.
- Follow the Paper2Video separation: render slides, narration, cursor, and subtitles first; then add presenter media from reference image/audio.
- Use the external talking-head hook only when `AIMOOC_TALKING_HEAD_CMD` or `talking_head_command` is configured; otherwise fallback to presenter-card overlay.
- Use calm academic presentation defaults unless course spec asks otherwise.
- Write `avatar_config.json` and per-lesson `avatar_manifest.json`.
