# SourceIngestionAgent Skills

- Accept multiple uploaded course sources without replacing the single-paper pipeline.
- Preserve per-source role, priority, title, and notes in `source_manifest.json`.
- Mark one primary source when the user only provides source IDs.
- Reject missing local files before planning.
- For video generation, merge selected readable PDFs into `source_bundle.pdf` so OCR, planning, slides, TTS, cursor, subtitles, and MP4 composition run over the full source set.
