# Agentic Automating Scientific Storytelling into Presentation Videos

A local control room that turns an uploaded scientific paper into an academic presentation video. The real pipeline uses MinerU OCR, local Ollama planning, Beamer slides, F5TTS narration, cursor grounding, subtitles, and ffmpeg MP4 composition.

## Features

- Upload paper PDFs from the browser.
- Queue real jobs and inspect job state, task history, and event replay.
- Select local Ollama text and vision models from a dropdown populated from `http://127.0.0.1:11434/api/tags`.
- Configure temperature, top-p, max tokens, system prompt, and local Ollama URL.
- Choose slide style templates with visual previews: Clean Academic, Dense Methods, Visual Results, and Teaching Walkthrough.
- Run MinerU OCR and expose extracted figures, tables, charts, code blocks, and formulas in the UI.
- Generate Beamer slides with OCR-grounded visual assets.
- Generate slide narration, subtitles, cursor paths, per-slide speech, and final MP4.
- Inspect every agent and tool used by the runtime.
- Open and edit each agent `skills.md` from the web UI.
- Inspect the LangGraph-backed agent handoff graph at `/api/agent-graph`.

## Agents

- `IngestionAgent`: receives uploaded PDFs, reads the upload manifest, runs MinerU OCR routing, and builds the OCR asset manifest used by slides and inspection.
- `PlannerAgent`: summarizes OCR content, allocates slide budget, calls the local Ollama text model, and prepares the talk structure.
- `SlideBuilderAgent`: writes Beamer frames, grounds OCR figures/tables/formulas into slides, and renders slide images.
- `ScriptAgent`: builds speaker notes, subtitle chunks, pacing hints, and cursor beat anchors.
- `SpeechAgent`: prepares reference voice metadata, stages F5TTS jobs, and generates per-slide narration audio.
- `GroundingAgent`: reads slide regions, aligns narration beats to visual focus targets, and writes cursor routes.
- `RenderAgent`: packages slide images, audio, subtitles, cursor overlay, and final MP4 artifacts through ffmpeg.

Each agent has an editable skills file under `src/agents/*/skills.md`. The web UI `Runtime capabilities` panel has `Open` and `Edit skills.md` controls for each agent.

## Agentic Graph

The control room builds an explicit LangGraph state graph from `src/agents/manifest.json` and `src/tools/manifest.json`:

```text
SupervisorAgent
  -> IngestionAgent
  -> PlannerAgent
  -> [SlideBuilderAgent -> VisualAuditorAgent] and [ScriptAgent -> SpeechAgent]
  -> ArtifactJoinAgent
  -> GroundingAgent
  -> RenderAgent
```

The graph is exposed by `/api/agent-graph`, included in `/api/health`, and rendered in the Run page `Agentic graph` panel with node edges and tool-call trace. It uses a supervisor, conditional routes, parallel fanout, join gating, and repair cycles. Each graph node owns its declared skills and tools. The real pipeline subprocess still performs the heavy OCR, model, TTS, cursor, and ffmpeg work; LangGraph provides the inspectable agent handoff contract in the web orchestration layer.

## Tools

The runtime tool registry lives in `src/tools/manifest.json`. Current tools include PDF manifest reading, MinerU OCR routing, OCR asset normalization, section planning, Ollama dispatch, Beamer writing, figure grounding, subtitle alignment, F5TTS queueing, cursor routing, slide rendering, ffmpeg packaging, and MP4 verification.

## Runtime Controls

`Runtime Controls` are count budgets used by the web monitor for visible agent/tool progress. Raising a count keeps that stage open longer in the monitor and task replay. It does not force real OCR, Ollama, TTS, LaTeX, or ffmpeg to take that exact amount of time. Real runtime depends on GPU, document length, model speed, OCR complexity, and speech duration.

The requested `Target length (minutes)` controls desired talk length and slide budget. The planner currently maps this to a bounded academic deck size, then the narration stage expands or trims speaker text. Actual final video length may still differ because F5TTS speech speed and slide count determine the final MP4 duration.

## Runtime Layout

- `web/app.py`: FastAPI control room, queue, task state, artifacts, settings, agent/skill APIs, and web routes.
- `web/static/`: browser UI for upload, settings, model/style selectors, history, replay, agents, tools, OCR assets, and output preview.
- `web/test_api.py`: backend and API smoke tests.
- `src/real_pipeline.py`: real OCR-to-video pipeline.
- `src/cursor_router.py`: deterministic cursor route generation through `build_cursor_timeline`.
- `src/cursor_overlay.py`: cursor overlay rendering through `render_cursor_overlay_timeline`.
- `src/speech_synth.py`: F5TTS per-slide speech generation through `synthesize_slide_audio`.
- `src/agents/`: agent catalog and per-agent `skills.md` files.
- `src/tools/manifest.json`: runtime tool registry.

Runtime/generated data is ignored by git:

- `result/`
- `web/data/uploads/`
- `web/data/tasks/`
- `web/server*.log`
- local voice samples under `assets/demo/`

## Requirements

Install these on the local machine:

- Python 3.12
- CUDA-capable PyTorch for GPU OCR/TTS/cursor workloads
- MinerU CLI
- Ollama running locally at `http://127.0.0.1:11434`
- Ollama model `qwen3.6:27b` or another selected local text model
- Optional local vision model listed by Ollama for future vision calls
- LangGraph for the inspectable agent handoff DAG
- F5TTS dependencies
- ffmpeg and ffprobe
- LaTeX distribution with `pdflatex`

The web worker chooses the pipeline Python in this order:

1. `PIPELINE_RUNTIME_PYTHON` environment variable, if set.
2. Repository `.runtime_env\Scripts\python.exe`, if present.
3. Repository `.venv\Scripts\python.exe`, if present.
4. The Python executable that started FastAPI.

For real TTS runs, use a Python environment with `whisperx` and `f5_tts` installed. `whisperx` is only needed when `--ref_text` is missing; F5TTS is needed for speech synthesis.

The preferred reference voice path is:

```text
assets/demo/reference.wav
```

If that file is missing, `src/real_pipeline.py` first uses the bundled F5TTS `basic_ref_en.wav` with its matching clean transcript, `Some call me nature, others call me mother nature.` If the bundled reference is unavailable, the pipeline creates a neutral `reference_fallback.wav` inside the job result directory so F5TTS does not fail with `FileNotFoundError`. Contaminated demo wording is filtered before synthesis. Voice quality is better when a real reference wav and matching transcript are provided.

Cursor overlay does not require a repository image asset. `src/cursor_overlay.py` draws the cursor directly with an ffmpeg `drawbox` filter from `cursor.json`.

The requested duration is treated as a pacing target, not a hard fill requirement; outputs may finish up to about one minute short to preserve natural speech. A LangGraph pacing planner chooses final slide count, content-slide budget, per-slide word budget, F5TTS speed policy, and subtitle source. If the user selects 12 slides, the pipeline targets 12 final Beamer pages including title and roadmap. F5TTS now runs at natural speed with one synthesis call per slide, matching the earlier stable voice behavior and avoiding artificial pauses around punctuation, hyphenated terms, or acronyms. Duration is controlled by agent-assigned script budget instead of slowing the synthesizer or padding silence. ScriptAgent expands short narration with varied academic bridge sentences and avoids repeated checklist openers. Burned-in subtitles are generated from a WhisperX ASR pass over the generated audio when available, then widened and held through each slide's narration window so spoken words and visible subtitles stay aligned.

## Run Web UI

```powershell
$env:PIPELINE_RUNTIME_PYTHON = ".\.runtime_env\Scripts\python.exe"
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
.\.venv\Scripts\python.exe -m py_compile src\real_pipeline.py src\cursor_router.py src\cursor_overlay.py src\speech_synth.py web\app.py web\test_api.py
.\.venv\Scripts\python.exe web\test_api.py
```

`web/test_api.py` creates temporary fixture artifacts under ignored runtime directories and removes them after success. It also verifies GPU availability, settings, model/style APIs, skill editing roundtrip, OCR assets, artifacts, task history, replay, and TTS reference fallback creation.

## Authorship

Repository commits should use only:

```text
fader2077 <fader2077.ai14@nycu.edu.tw>
```
