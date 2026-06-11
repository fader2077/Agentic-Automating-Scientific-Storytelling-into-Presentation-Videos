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
