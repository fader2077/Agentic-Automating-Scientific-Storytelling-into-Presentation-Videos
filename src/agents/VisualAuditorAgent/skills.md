# VisualAuditorAgent Skills

## Responsibilities
- Check whether slides use extracted OCR visuals when available.
- Verify formulas, tables, and figures are grounded into renderable slide regions.
- Route slide repair when visual coverage or rendering is weak.

## Skills
- Visual coverage audit
- Formula and table placement checks
- Slide repair routing

## Tools
- `figure_grounder`
- `slide_renderer`

## Runtime Inputs
- `ocr_assets.json`
- `latex_proj/slides.pdf`
- `slide_imgs/*.png`

## Runtime Outputs
- Visual audit state
- Repair or join route

## Agentic Policy
- Enter after `SlideBuilderAgent`.
- Route back to `SlideBuilderAgent` when visual evidence is missing or slide rendering fails.
- Publish audit completion before artifacts can join with speech output.
