# IngestionAgent Skills

## Responsibilities
- Validate uploaded paper files and retain the saved source path.
- Run MinerU OCR with the pipeline backend.
- Collect OCR markdown, content-list JSON, layout PDFs, page images, and region metadata.
- Build an OCR asset manifest for images, charts, tables, equations, and code blocks.

## Skills
- PDF intake
- MinerU OCR
- Layout inventory
- OCR asset extraction

## Tools
- `pdf_manifest`
- `ocr_router`
- `ocr_asset_manifest`

## Runtime Inputs
- Uploaded PDF path
- Job result directory
- MinerU method, usually `ocr`

## Runtime Outputs
- `mineru/.../*.md`
- `mineru/.../*content_list*.json`
- `ocr_assets.json`
- Copied visual assets under `latex_proj/ocr_assets`

## Agentic Policy
- Enter from `SupervisorAgent` when no reusable OCR artifacts exist.
- Decide whether MinerU output is usable by checking markdown, content-list JSON, and visual asset counts.
- If OCR output is missing or malformed, hand off to `IngestionRepairAgent` for one repair/retry cycle.
- If OCR assets are valid, publish `ocr_assets` state and hand off to `PlannerAgent`.
- Never silently fabricate paper content; failed OCR must remain an explicit graph error.
