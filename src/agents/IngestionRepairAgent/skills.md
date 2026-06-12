# IngestionRepairAgent Skills

## Responsibilities
- Repair failed OCR or malformed OCR manifests.
- Decide whether to retry MinerU or rebuild the asset manifest.
- Return control to `IngestionAgent` after repair.

## Skills
- OCR retry
- Manifest repair
- Failure triage

## Tools
- `ocr_router`
- `ocr_asset_manifest`

## Runtime Inputs
- OCR failure state
- MinerU output directory
- Partial markdown or content-list JSON

## Runtime Outputs
- Repaired OCR artifacts
- Retry decision

## Agentic Policy
- Run only on conditional error routes.
- Never fabricate OCR assets.
- Limit repair loops so the graph fails explicitly if OCR remains invalid.
