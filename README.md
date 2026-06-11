# Agentic Automating Scientific Storytelling into Presentation Videos

A local research-video control room that turns an uploaded paper into an academic Beamer deck, narration, subtitles, cursor grounding, and final MP4. The system exposes a web UI plus a real pipeline backed by MinerU OCR, local Ollama planning, F5TTS, and ffmpeg.

## Features

- Upload paper PDFs from the web UI.
- Run a queue-backed job pipeline with visible agent, skill, and tool state.
- Use MinerU to extract markdown, equations, tables, figures, and layout assets.
- Plan slide structure with a local Ollama model such as `qwen3.6:27b`.
- Generate Beamer slides with OCR-grounded visual assets.
- Create slide-level narration, subtitles, cursor paths, and MP4 output.
- Inspect task history and replay pipeline events.

## Runtime Layout

Core source files:

- `web/app.py` - FastAPI control room, queue, task state, artifacts, settings, and web routes.
- `web/static/` - browser UI for upload, settings, history, replay, agents, and tools.
- `src/real_pipeline.py` - real OCR-to-video pipeline.
- `src/cursor_gen.py` - deterministic cursor route generation.
- `src/cursor_render.py` - cursor overlay rendering.
- `src/speech_gen.py` - F5TTS per-slide speech generation.
- `src/agents/` - agent catalog and per-agent `skills.md` files.
- `src/tools/manifest.json` - runtime tool registry.

Runtime/generated data is intentionally ignored by git:

- `result/`
- `web/data/uploads/`
- `web/data/tasks/`
- `web/server*.log`
- local voice samples under `assets/demo/`

## Requirements

Install these outside the repository as appropriate for your machine:

- Python 3.12
- CUDA-capable PyTorch if using GPU TTS/OCR
- MinerU CLI
- Ollama running locally at `http://127.0.0.1:11434`
- Ollama model `qwen3.6:27b` or another configured text model
- F5TTS dependencies
- ffmpeg and ffprobe
- LaTeX distribution with `pdflatex`

The current code expects a reference voice by default at:

```text
assets/demo/reference.wav
```

You can also call `src/real_pipeline.py` directly with `--ref_audio` and `--ref_text`.

## Run Web UI

```powershell
.\.venv\Scripts\python.exe -m uvicorn web.app:app --host 127.0.0.1 --port 8008
```

Open:

```text
http://127.0.0.1:8008
```

## Run Pipeline Directly

```powershell
.\.venv\Scripts\python.exe src\real_pipeline.py `
  --paper_pdf path\to\paper.pdf `
  --result_dir result\job_manual `
  --desired_minutes 3 `
  --goal_prompt "Create a rigorous academic presentation." `
  --model qwen3.6:27b `
  --ollama_url http://127.0.0.1:11434 `
  --temperature 0.2 `
  --top_p 0.9 `
  --mineru_method ocr `
  --ref_audio assets\demo\reference.wav `
  --ref_text "Reference speaker transcript."
```

## Test

```powershell
.\.venv\Scripts\python.exe -m py_compile src\real_pipeline.py src\cursor_gen.py src\cursor_render.py src\speech_gen.py web\app.py web\test_api.py
.\.venv\Scripts\python.exe web\test_api.py
```

`web/test_api.py` creates temporary fixture artifacts under ignored runtime directories and removes them after success.

## Authorship

Repository commits should use:

```text
fader2077 <fader2077.ai14@nycu.edu.tw>
```

