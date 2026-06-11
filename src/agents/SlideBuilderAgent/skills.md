# SlideBuilderAgent Skills

## Responsibilities
- Convert the model plan into Beamer LaTeX.
- Ground slides with MinerU OCR assets when available.
- Compile Beamer with `pdflatex`.
- Render slide PDF pages to PNG images for video composition.

## Skills
- Beamer generation
- OCR visual grounding
- Table placement
- Figure placement

## Tools
- `beamer_writer`
- `figure_grounder`
- `slide_renderer`

## Runtime Inputs
- `plan.json`
- `ocr_assets.json`
- LaTeX output directory

## Runtime Outputs
- `latex_proj/slides.tex`
- `latex_proj/slides.pdf`
- `slide_imgs/*.png`
