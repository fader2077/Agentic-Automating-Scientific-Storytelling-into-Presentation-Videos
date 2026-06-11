# PlannerAgent Skills

## Responsibilities
- Compact OCR markdown into a model-sized planning context.
- Call local Ollama through `qwen3.6:27b`.
- Produce a structured JSON talk plan with title, authors, summary, slide bullets, speaker text, and cursor hints.
- Keep generated claims tied to the OCR text.

## Skills
- Paper summarization
- Talk pacing
- Section allocation
- Local Ollama planning

## Tools
- `section_planner`
- `ollama_dispatch`

## Runtime Inputs
- OCR markdown
- User goal prompt
- Desired presentation length
- Ollama URL, model, temperature, and top-p settings

## Runtime Outputs
- `plan.json`
- `ollama_plan_raw.txt`
