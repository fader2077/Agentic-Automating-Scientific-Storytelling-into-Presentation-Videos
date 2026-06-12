from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import importlib.util
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

import fitz

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cursor_router import build_cursor_timeline
from cursor_overlay import render_cursor_overlay_timeline
from speech_synth import synthesize_slide_audio


DEFAULT_MODEL = "qwen3.6:27b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
F5_BASIC_REF_TEXT = "Some call me nature, others call me mother nature."


class PipelineError(RuntimeError):
    pass


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise PipelineError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\nSTDOUT:\n"
            + result.stdout[-4000:]
            + "\nSTDERR:\n"
            + result.stderr[-4000:]
        )
    return result


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def write_fallback_reference_audio(path: Path, duration_seconds: float = 2.4, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_samples = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(total_samples):
            sample = int(0.08 * 32767 * math.sin(2 * math.pi * 220 * index / sample_rate))
            frames.extend(struct.pack("<h", sample))
        handle.writeframes(bytes(frames))


def ensure_reference_audio(ref_audio: str, result_dir: Path) -> Path:
    candidate = Path(ref_audio)
    if candidate.exists():
        return candidate
    package_ref = find_f5_reference_audio()
    if package_ref is not None:
        return package_ref
    fallback = result_dir / "reference_fallback.wav"
    if not fallback.exists():
        write_fallback_reference_audio(fallback)
    return fallback


def find_f5_reference_audio() -> Path | None:
    try:
        spec = importlib.util.find_spec("f5_tts")
    except (ModuleNotFoundError, ValueError):
        return None
    if not spec or not spec.submodule_search_locations:
        return None
    package_root = Path(next(iter(spec.submodule_search_locations)))
    candidate = package_root / "infer" / "examples" / "basic" / "basic_ref_en.wav"
    return candidate if candidate.exists() else None


def resolve_reference_voice(ref_audio: str, ref_text: str | None, result_dir: Path) -> tuple[Path, str | None]:
    requested = Path(ref_audio)
    if requested.exists():
        return requested, ref_text
    package_ref = find_f5_reference_audio()
    if package_ref is not None:
        return package_ref, ref_text or F5_BASIC_REF_TEXT
    fallback = ensure_reference_audio(ref_audio, result_dir)
    return fallback, ref_text or "Reference voice for generated academic narration."


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def emit_event(kind: str, step: str, message: str, data: dict[str, Any] | None = None) -> None:
    payload = {
        "kind": kind,
        "step": step,
        "message": message,
        "data": data or {},
        "timestamp": time.time(),
    }
    print("PIPELINE_EVENT " + json.dumps(payload, ensure_ascii=False), flush=True)


def ensure_ollama_model(model: str, base_url: str) -> None:
    tags_url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise PipelineError(f"Ollama is not reachable at {base_url}: {exc}") from exc

    models = {item.get("name") for item in payload.get("models", [])}
    if model not in models:
        raise PipelineError(f"Required Ollama model is missing: {model}. Available: {sorted(models)}")


def ollama_generate(
    prompt: str,
    model: str,
    base_url: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    timeout: int = 900,
) -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": 8192,
        },
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            chunks: list[str] = []
            thinking_chunks: list[str] = []
            raw_lines: list[str] = []
            for line in response:
                decoded = line.decode("utf-8", errors="ignore").strip()
                if not decoded:
                    continue
                raw_lines.append(decoded)
                payload = json.loads(decoded)
                chunks.append(payload.get("response", ""))
                thinking_chunks.append(payload.get("thinking", ""))
    except urllib.error.HTTPError as exc:
        raise PipelineError(f"Ollama HTTP error: {exc.read().decode('utf-8', errors='ignore')}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Ollama returned invalid JSON line: {exc}") from exc
    except Exception as exc:
        raise PipelineError(f"Ollama generation failed: {exc}") from exc
    text = "".join(chunks)
    if not text.strip():
        text = "".join(thinking_chunks)
    if not text.strip():
        diagnostics = "\n".join(raw_lines[-5:])
        raise PipelineError(f"Ollama returned an empty response. Last payload lines:\n{diagnostics}")
    return text


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise PipelineError("No JSON object found in model response.")
    raw = text[start : end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Invalid JSON from model: {exc}\n{raw[:2000]}") from exc


def run_mineru(pdf_path: Path, output_dir: Path, method: str = "ocr") -> tuple[Path, Path]:
    existing_md = sorted(output_dir.rglob("*.md")) if output_dir.exists() else []
    if existing_md:
        json_files = sorted(output_dir.rglob("*content_list*.json"))
        return existing_md[0], json_files[0] if json_files else existing_md[0]

    clean_dir(output_dir)
    run(
        [
            "mineru",
            "-p",
            str(pdf_path),
            "-o",
            str(output_dir),
            "-m",
            method,
            "-b",
            "pipeline",
            "-l",
            "en",
        ],
        timeout=3600,
    )
    md_files = sorted(output_dir.rglob("*.md"))
    json_files = sorted(output_dir.rglob("*content_list*.json"))
    if not md_files:
        raise PipelineError(f"MinerU did not produce markdown under {output_dir}")
    return md_files[0], json_files[0] if json_files else md_files[0]


def plain_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(plain_text(item) for item in value)
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


UNICODE_LATEX_MAP = {
    "\u00a0": " ",
    "\u03b1": "alpha",
    "\u03b2": "beta",
    "\u03b3": "gamma",
    "\u03b4": "delta",
    "\u03bb": "lambda",
    "\u03bc": "mu",
    "\u03c3": "sigma",
    "\u03c4": "tau",
    "\u03c6": "phi",
    "\u03c8": "psi",
    "\u03c9": "omega",
    "\u0394": "Delta",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "--",
    "\u2014": "---",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
}


def normalize_latex_text(value: Any) -> str:
    text = plain_text(value)
    for source, target in UNICODE_LATEX_MAP.items():
        text = text.replace(source, target)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def audit_asset_caption(caption: str, kind: str, page: int) -> str:
    caption = normalize_latex_text(caption)
    if not caption:
        return f"OCR {kind} from page {page}"

    cleaned = re.sub(r"[^A-Za-z0-9\s.,;:()/%+\-=]", "", caption)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return f"OCR {kind} from page {page}"

    original_len = max(len(caption), 1)
    retained_ratio = len(cleaned) / original_len
    punct_ratio = len(re.findall(r"[^A-Za-z0-9\s]", cleaned)) / max(len(cleaned), 1)
    words = cleaned.split()

    if retained_ratio < 0.68 or punct_ratio > 0.22:
        return f"OCR {kind} from page {page}"

    if len(words) > 26:
        cleaned = " ".join(words[:26]).rstrip(".,;:") + "."
    elif len(cleaned) > 170:
        cleaned = cleaned[:170].rsplit(" ", 1)[0].rstrip(".,;:") + "."

    if len(cleaned) < 8:
        return f"OCR {kind} from page {page}"
    return cleaned


def build_ocr_assets(content_json: Path, latex_dir: Path, manifest_path: Path) -> list[dict[str, Any]]:
    if not content_json.exists() or content_json.suffix.lower() != ".json":
        write_json(manifest_path, {"assets": [], "counts": {}})
        return []

    data = json.loads(content_json.read_text(encoding="utf-8"))
    source_root = content_json.parent
    asset_dir = latex_dir / "ocr_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    supported = {"image", "table", "chart", "code", "equation", "interline_equation"}

    for item in data:
        kind = str(item.get("type", "")).strip()
        if kind not in supported:
            continue
        counts[kind] = counts.get(kind, 0) + 1
        raw_caption = (
            item.get("image_caption")
            or item.get("table_caption")
            or item.get("chart_caption")
            or item.get("code_caption")
            or item.get("text")
            or item.get("table_body")
            or item.get("code_body")
            or item.get("content")
        )
        caption = audit_asset_caption(raw_caption, kind, int(item.get("page_idx", 0)) + 1)
        image_path = item.get("img_path")
        copied_path = ""
        if image_path:
            source_path = (source_root / str(image_path)).resolve()
            if source_path.exists():
                suffix = source_path.suffix.lower() or ".jpg"
                copied_path = str(asset_dir / f"{kind}_{counts[kind]:02d}{suffix}")
                shutil.copyfile(source_path, copied_path)
        assets.append(
            {
                "id": f"{kind}_{counts[kind]:02d}",
                "kind": kind,
                "page": int(item.get("page_idx", 0)) + 1,
                "caption": caption[:500],
                "image": copied_path,
                "body": normalize_latex_text(item.get("table_body") or item.get("code_body") or item.get("content"))[:1000],
            }
        )

    manifest = {
        "source": str(content_json),
        "asset_dir": str(asset_dir),
        "counts": counts,
        "assets": assets,
    }
    write_json(manifest_path, manifest)
    return assets


def compact_markdown(markdown: str, max_chars: int = 42000) -> str:
    markdown = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = markdown.strip()
    if len(markdown) <= max_chars:
        return markdown
    head = markdown[: int(max_chars * 0.72)]
    tail = markdown[-int(max_chars * 0.28) :]
    return head + "\n\n[... middle content omitted for context budget ...]\n\n" + tail


def build_deck_prompt(markdown: str, goal_prompt: str, desired_minutes: int) -> str:
    slide_count = target_content_slide_count(desired_minutes)
    return f"""
/no_think
You are generating a real academic presentation-video plan from OCR text.
Return ONLY valid JSON. No markdown fences.

Target:
- {slide_count} slides.
- Academic, concise, faithful to the paper.
- Beamer slides will be produced from your JSON.
- Narration must be English, specific, and paced for the requested video length.
- Every slide needs a cursor_hint for visual focus.

User instruction:
{goal_prompt}

JSON schema:
{{
  "title": "paper title",
  "authors": "author names or empty string",
  "one_sentence_summary": "faithful summary",
  "slides": [
    {{
      "title": "slide title",
      "bullets": ["bullet 1", "bullet 2", "bullet 3"],
      "speaker": "one or two concise narration sentences, paced for the requested video length",
      "cursor_hint": "title area | main bullet list | figure area | table area | center of slide"
    }}
  ]
}}

Hard constraints:
- Use {slide_count} slides exactly.
- Each slide has 2 to 4 bullets.
- Each bullet must be under 18 words.
- Each speaker field must be 1 to 2 sentences and stay concise; the pipeline will enforce the final word budget.
- No invented numeric results unless present in the paper text.
- Avoid citations and bibliography slides.
- Avoid appendix/checklist content.

OCR text:
{compact_markdown(markdown, max_chars=14000)}
""".strip()


def target_content_slide_count(desired_minutes: int) -> int:
    return max(8, min(22, int(desired_minutes) * 4))


def target_speaker_words(desired_minutes: int) -> int:
    slide_count = target_content_slide_count(desired_minutes)
    target_total_words = max(160, int(desired_minutes) * 125)
    rendered_slide_count = slide_count + 1
    return max(26, min(48, target_total_words // max(rendered_slide_count, 1)))


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def trim_speaker_text(speaker: str, max_words: int) -> str:
    speaker = re.sub(r"\s+", " ", speaker).strip()
    if word_count(speaker) <= max_words:
        return speaker

    sentences = re.split(r"(?<=[.!?])\s+", speaker)
    kept: list[str] = []
    for sentence in sentences:
        candidate = " ".join(kept + [sentence]).strip()
        if candidate and word_count(candidate) <= max_words:
            kept.append(sentence)
        elif kept:
            break
    if kept:
        return " ".join(kept).strip()

    words = re.findall(r"\S+", speaker)
    trimmed = " ".join(words[:max_words]).rstrip(".,;:")
    if trimmed and not trimmed.endswith((".", "!", "?")):
        trimmed += "."
    return trimmed


def expand_speaker_text(speaker: str, slide_title: str, bullets: list[str], desired_minutes: int) -> str:
    speaker = re.sub(r"\s+", " ", speaker).strip()
    if not speaker:
        speaker = f"This slide explains {slide_title}."
    if not speaker.endswith((".", "!", "?")):
        speaker += "."

    target_words = target_speaker_words(desired_minutes)
    if word_count(speaker) >= target_words:
        return trim_speaker_text(speaker, target_words + 5)

    additions = [
        "The visual evidence anchors this point in the extracted paper content.",
        "This also prepares the evaluation and limitations discussed later.",
    ]
    for addition in additions:
        if word_count(speaker) >= target_words:
            break
        speaker = f"{speaker} {addition}"
    return trim_speaker_text(speaker, target_words + 5)


def validate_plan(plan: dict[str, Any], desired_minutes: int) -> dict[str, Any]:
    title = str(plan.get("title") or "Academic Paper Presentation").strip()
    authors = str(plan.get("authors") or "").strip()
    slides = plan.get("slides")
    if not isinstance(slides, list) or len(slides) < 5:
        raise PipelineError("Model plan has too few slides.")

    max_slides = target_content_slide_count(desired_minutes)
    slides = slides[:max_slides]
    normalized = []
    for idx, slide in enumerate(slides, start=1):
        if not isinstance(slide, dict):
            continue
        slide_title = str(slide.get("title") or f"Slide {idx}").strip()
        bullets_raw = slide.get("bullets") or []
        bullets = [str(item).strip() for item in bullets_raw if str(item).strip()]
        bullets = bullets[:4]
        if len(bullets) < 2:
            bullets.extend(["Key idea from the paper", "Evidence and implication"])
        speaker = str(slide.get("speaker") or f"This slide explains {slide_title}.").strip()
        speaker = expand_speaker_text(speaker, slide_title, bullets, desired_minutes)
        cursor_hint = str(slide.get("cursor_hint") or "main bullet list").strip()
        normalized.append(
            {
                "title": slide_title[:90],
                "bullets": [b[:150] for b in bullets],
                "speaker": speaker[:900],
                "cursor_hint": cursor_hint[:80],
            }
        )
    if len(normalized) < 5:
        raise PipelineError("Model plan normalization produced too few slides.")
    return {
        "title": title[:140],
        "authors": authors[:180],
        "one_sentence_summary": str(plan.get("one_sentence_summary") or "").strip()[:300],
        "slides": normalized,
    }


def tex_escape(text: str) -> str:
    text = normalize_latex_text(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def select_visual_asset(slide: dict[str, Any], assets: list[dict[str, Any]], used_ids: set[str]) -> dict[str, Any] | None:
    if not assets:
        return None
    text = f"{slide.get('title', '')} {' '.join(slide.get('bullets', []))}".lower()
    preferred: list[str]
    if any(term in text for term in ["benchmark", "dataset", "comparison"]):
        preferred = ["table", "chart", "image"]
    elif any(term in text for term in ["metric", "evaluation", "experiment", "result"]):
        preferred = ["table", "chart", "equation", "image"]
    elif any(term in text for term in ["method", "model", "pipeline", "system", "framework", "retrieval", "rag", "graph", "score", "loss", "objective", "confidence", "equation", "formula"]):
        preferred = ["equation", "image", "code", "chart", "table"]
    elif any(term in text for term in ["visual", "layout", "tree search", "slide"]):
        preferred = ["image", "chart", "code"]
    else:
        preferred = ["image", "chart", "table", "equation", "code"]

    for kind in preferred:
        for asset in assets:
            if asset["id"] not in used_ids and asset["kind"] == kind and asset.get("image"):
                used_ids.add(asset["id"])
                return asset
    for asset in assets:
        if asset["id"] not in used_ids and asset.get("image"):
            used_ids.add(asset["id"])
            return asset
    return None


def assign_visual_assets(slides: list[dict[str, Any]], assets: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    used_assets: set[str] = set()
    selected = [select_visual_asset(slide, assets, used_assets) for slide in slides]
    if any(asset and asset.get("kind") == "equation" for asset in selected):
        return selected

    equation = next((asset for asset in assets if asset.get("kind") == "equation" and asset.get("image")), None)
    if not equation:
        return selected

    preferred_terms = ["method", "model", "retrieval", "rag", "graph", "score", "loss", "objective", "confidence"]
    replacement_index = 0
    for index, slide in enumerate(slides):
        text = f"{slide.get('title', '')} {' '.join(slide.get('bullets', []))}".lower()
        if any(term in text for term in preferred_terms):
            replacement_index = index
            break
    selected[replacement_index] = equation
    return selected


def latex_image_path(path_text: str) -> str:
    return Path(path_text).resolve().as_posix()


def write_beamer(plan: dict[str, Any], tex_path: Path, ocr_assets: list[dict[str, Any]] | None = None) -> None:
    slides = plan["slides"]
    ocr_assets = ocr_assets or []
    assigned_assets = assign_visual_assets(slides, ocr_assets)
    lines = [
        r"\documentclass[aspectratio=169]{beamer}",
        r"\usetheme{Madrid}",
        r"\usecolortheme{default}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\setbeamertemplate{navigation symbols}{}",
        r"\setbeamertemplate{footline}[frame number]",
        r"\setbeamerfont{frametitle}{size=\large}",
        r"\setbeamerfont{itemize/enumerate body}{size=\small}",
        f"\\title{{{tex_escape(plan['title'])}}}",
        f"\\author{{{tex_escape(plan.get('authors') or 'Generated from paper OCR')}}}",
        r"\date{}",
        r"\begin{document}",
        r"\begin{frame}",
        r"\titlepage",
        r"\end{frame}",
    ]
    if plan.get("one_sentence_summary"):
        lines.extend(
            [
                r"\begin{frame}{Talk Roadmap}",
                r"\small",
                f"\\textbf{{Summary.}} {tex_escape(plan['one_sentence_summary'])}",
                r"\vspace{0.6em}",
                r"\begin{itemize}",
                r"\item Motivation and problem setting",
                r"\item Method and system design",
                r"\item Experiments, findings, and limitations",
                r"\end{itemize}",
                r"\end{frame}",
            ]
        )
    for slide, asset in zip(slides, assigned_assets):
        lines.extend(
            [
                f"\\begin{{frame}}{{{tex_escape(slide['title'])}}}",
                r"\small",
            ]
        )
        if asset:
            lines.extend(
                [
                    r"\begin{columns}[T,totalwidth=\textwidth]",
                    r"\begin{column}{0.47\textwidth}",
                    r"\begin{itemize}",
                ]
            )
        else:
            lines.append(r"\begin{itemize}")
        for bullet in slide["bullets"]:
            lines.append(f"\\item {tex_escape(bullet)}")
        if asset:
            caption = asset.get("caption") or f"OCR {asset['kind']} from page {asset['page']}"
            if asset.get("kind") == "equation":
                asset_note = f"{{\\scriptsize\\textbf{{OCR Equation, p.{asset['page']}.}}}}"
            else:
                asset_note = f"{{\\scriptsize\\textbf{{OCR {tex_escape(asset['kind'].title())}, p.{asset['page']}.}} {tex_escape(caption[:180])}}}"
            lines.extend(
                [
                    r"\end{itemize}",
                    r"\end{column}",
                    r"\begin{column}{0.50\textwidth}",
                    r"\centering",
                    f"\\includegraphics[width=\\linewidth,height=0.53\\textheight,keepaspectratio]{{{latex_image_path(asset['image'])}}}",
                    r"\vspace{0.2em}",
                    asset_note,
                    r"\end{column}",
                    r"\end{columns}",
                    r"\end{frame}",
                ]
            )
        else:
            lines.extend([r"\end{itemize}", r"\end{frame}"])
    lines.append(r"\end{document}")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_beamer(tex_path: Path) -> Path:
    for _ in range(2):
        run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=tex_path.parent,
            timeout=180,
        )
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise PipelineError(f"Beamer PDF missing after compile: {pdf_path}")
    return pdf_path


def render_slide_images(pdf_path: Path, slide_img_dir: Path) -> int:
    clean_dir(slide_img_dir)
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            pix.save(str(slide_img_dir / f"{index}.png"))
        return doc.page_count


def write_script(plan: dict[str, Any], script_path: Path) -> None:
    pages = []
    pages.append(
        "\n".join(
            [
                f"Today I present {plan['title']}. | title area",
                "I will summarize the paper's motivation, method, evidence, and limitations. | center of slide",
            ]
        )
    )
    if plan.get("one_sentence_summary"):
        pages.append(f"{plan['one_sentence_summary']} | main bullet list")
    for slide in plan["slides"]:
        speaker = re.sub(r"\s+", " ", slide["speaker"]).strip()
        if not speaker.endswith((".", "!", "?")):
            speaker += "."
        pages.append(f"{speaker} | {slide['cursor_hint']}")
    script_path.write_text("\n###\n".join(pages) + "\n", encoding="utf-8")


def ffprobe_duration(path: Path) -> float:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        timeout=30,
    )
    return float(result.stdout.strip())


def numeric_wav_files(audio_dir: Path) -> list[Path]:
    return sorted(audio_dir.glob("*.wav"), key=lambda item: int(item.stem) if item.stem.isdigit() else item.stem)


def atempo_filter(speed: float) -> str:
    factors: list[float] = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6g}" for factor in factors)


def normalize_audio_total_duration(audio_dir: Path, desired_minutes: int) -> dict[str, Any]:
    files = numeric_wav_files(audio_dir)
    durations = [ffprobe_duration(path) for path in files]
    original_total = sum(durations)
    target_total = max(30.0, float(desired_minutes) * 60.0)
    if not files or original_total <= 0:
        return {"changed": False, "original_total": round(original_total, 3), "target_total": round(target_total, 3)}

    speed = original_total / target_total
    if 0.95 <= speed <= 1.08:
        return {
            "changed": False,
            "original_total": round(original_total, 3),
            "target_total": round(target_total, 3),
            "speed": round(speed, 3),
        }

    audio_filter = atempo_filter(speed)
    for wav_path in files:
        temp_path = wav_path.with_suffix(".normalized.wav")
        run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(wav_path),
                "-filter:a",
                audio_filter,
                str(temp_path),
            ],
            timeout=180,
        )
        os.replace(temp_path, wav_path)
    normalized_total = sum(ffprobe_duration(path) for path in files)
    return {
        "changed": True,
        "original_total": round(original_total, 3),
        "target_total": round(target_total, 3),
        "normalized_total": round(normalized_total, 3),
        "speed": round(speed, 3),
        "filter": audio_filter,
    }


def build_srt_from_script(script_path: Path, audio_dir: Path, srt_path: Path) -> None:
    pages = [p.strip() for p in script_path.read_text(encoding="utf-8").split("###") if p.strip()]
    current = 0.0
    entries = []
    entry_idx = 1
    for slide_idx, page in enumerate(pages):
        wav = audio_dir / f"{slide_idx}.wav"
        duration = ffprobe_duration(wav)
        text_parts = []
        for line in page.splitlines():
            if "|" in line:
                text_parts.append(line.split("|", 1)[0].strip())
        text = " ".join(text_parts).strip()
        start = current
        end = current + duration
        entries.append((entry_idx, start, end, text))
        entry_idx += 1
        current = end

    def fmt(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        h, rem = divmod(ms, 3600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    content = []
    for idx, start, end, text in entries:
        content.extend([str(idx), f"{fmt(start)} --> {fmt(end)}", text, ""])
    srt_path.write_text("\n".join(content), encoding="utf-8")


def build_page_clips(result_dir: Path, slide_count: int) -> Path:
    list_path = result_dir / "concat_list.txt"
    list_path.write_text("", encoding="ascii")
    for slide_idx in range(1, slide_count + 1):
        slide_path = result_dir / "slide_imgs" / f"{slide_idx}.png"
        audio_path = result_dir / "audio" / f"{slide_idx - 1}.wav"
        out_path = result_dir / f"page_{slide_idx:03d}.mp4"
        duration = ffprobe_duration(audio_path)
        run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(slide_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-vf",
                "scale=1280:720,setsar=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "30",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-shortest",
                str(out_path),
            ],
            timeout=180,
        )
        with list_path.open("a", encoding="ascii") as handle:
            handle.write(f"file '{out_path.as_posix()}'\n")

    merged = result_dir / "1_merage.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(merged),
        ],
        timeout=300,
    )
    return merged


def burn_subtitles(video_in: Path, srt_path: Path, video_out: Path) -> None:
    style = (
        "FontName=Arial,FontSize=13,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BackColour=&H99000000,"
        "BorderStyle=4,Outline=1,Shadow=0,Alignment=2,MarginV=24"
    )
    sub_path = srt_path.resolve().as_posix().replace(":", "\\:").replace("'", "\\'")
    run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-i",
            str(video_in),
            "-vf",
            f"subtitles='{sub_path}':force_style='{style}'",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "copy",
            str(video_out),
        ],
        timeout=300,
    )


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    result_dir = Path(args.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "latex_proj").mkdir(parents=True, exist_ok=True)

    pdf_path = Path(args.paper_pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    ensure_ollama_model(args.model, args.ollama_url)

    metadata: dict[str, Any] = {
        "mode": "real",
        "model": args.model,
        "ollama_url": args.ollama_url,
        "paper_pdf": str(pdf_path),
        "steps": {},
    }

    t = time.time()
    emit_event("start", "mineru_ocr", "MinerU OCR started.", {"pdf": str(pdf_path)})
    md_path, content_json = run_mineru(pdf_path, result_dir / "mineru", method=args.mineru_method)
    markdown = read_text(md_path)
    ocr_assets = build_ocr_assets(content_json, result_dir / "latex_proj", result_dir / "ocr_assets.json")
    metadata["steps"]["mineru_ocr"] = {
        "seconds": round(time.time() - t, 3),
        "markdown": str(md_path),
        "content_json": str(content_json),
        "assets": len(ocr_assets),
    }
    emit_event("done", "mineru_ocr", "MinerU OCR completed.", metadata["steps"]["mineru_ocr"])

    t = time.time()
    emit_event("start", "ollama_plan", "Ollama slide planning started.", {"model": args.model})
    prompt = build_deck_prompt(markdown, args.goal_prompt, args.desired_minutes)
    raw_plan = ollama_generate(
        prompt,
        model=args.model,
        base_url=args.ollama_url,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    plan = validate_plan(extract_json(raw_plan), args.desired_minutes)
    write_json(result_dir / "plan.json", plan)
    (result_dir / "ollama_plan_raw.txt").write_text(raw_plan, encoding="utf-8")
    metadata["steps"]["ollama_plan"] = {"seconds": round(time.time() - t, 3), "slides": len(plan["slides"]) + 1}
    emit_event("done", "ollama_plan", "Ollama slide planning completed.", metadata["steps"]["ollama_plan"])

    t = time.time()
    emit_event("start", "beamer", "Beamer rendering started.", {})
    tex_path = result_dir / "latex_proj" / "slides.tex"
    write_beamer(plan, tex_path, ocr_assets=ocr_assets)
    pdf_out = compile_beamer(tex_path)
    slide_count = render_slide_images(pdf_out, result_dir / "slide_imgs")
    metadata["steps"]["beamer"] = {"seconds": round(time.time() - t, 3), "slides_pdf": str(pdf_out), "slide_count": slide_count}
    emit_event("done", "beamer", "Beamer rendering completed.", metadata["steps"]["beamer"])

    t = time.time()
    emit_event("start", "script", "Narration script generation started.", {})
    script_path = result_dir / "subtitle_w_cursor.txt"
    write_script(plan, script_path)
    metadata["steps"]["script"] = {"seconds": round(time.time() - t, 3), "script": str(script_path)}
    emit_event("done", "script", "Narration script generation completed.", metadata["steps"]["script"])

    t = time.time()
    emit_event("start", "tts", "F5TTS synthesis started.", {"engine": "F5TTS"})
    audio_dir = result_dir / "audio"
    clean_dir(audio_dir)
    ref_audio_path, ref_text = resolve_reference_voice(args.ref_audio, args.ref_text, result_dir)
    if str(ref_audio_path) != args.ref_audio:
        emit_event("info", "tts", "Reference audio missing; using fallback reference voice.", {"ref_audio": str(ref_audio_path)})
    synthesize_slide_audio(
        model_type="f5",
        script_path=str(script_path),
        speech_save_dir=str(audio_dir),
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
    )
    duration_fit = normalize_audio_total_duration(audio_dir, args.desired_minutes)
    metadata["steps"]["tts"] = {
        "seconds": round(time.time() - t, 3),
        "audio_files": len(list(audio_dir.glob("*.wav"))),
        "duration_fit": duration_fit,
    }
    emit_event("done", "tts", "F5TTS synthesis completed.", metadata["steps"]["tts"])

    t = time.time()
    emit_event("start", "cursor", "Cursor grounding started.", {})
    cursor_path = result_dir / "cursor.json"
    build_cursor_timeline(
        script_path=str(script_path),
        slide_img_dir=str(result_dir / "slide_imgs"),
        slide_audio_dir=str(audio_dir),
        cursor_save_path=str(cursor_path),
        gpu_list=[0],
    )
    metadata["steps"]["cursor"] = {"seconds": round(time.time() - t, 3), "cursor": str(cursor_path)}
    emit_event("done", "cursor", "Cursor grounding completed.", metadata["steps"]["cursor"])

    t = time.time()
    emit_event("start", "video", "Video composition started.", {})
    srt_path = result_dir / "subtitles.srt"
    build_srt_from_script(script_path, audio_dir, srt_path)
    merged = build_page_clips(result_dir, slide_count)
    with_cursor = result_dir / "2_merage.mp4"
    try:
        render_cursor_overlay_timeline(
            video_path=str(merged),
            out_video_path=str(with_cursor),
            json_path=str(cursor_path),
            transition_duration=0.1,
            cursor_size=14,
        )
    except UnicodeEncodeError:
        if not with_cursor.exists():
            raise
    final_video = result_dir / "3_merage.mp4"
    burn_subtitles(with_cursor, srt_path, final_video)
    duration = ffprobe_duration(final_video)
    metadata["steps"]["video"] = {"seconds": round(time.time() - t, 3), "final_video": str(final_video), "duration": round(duration, 3)}
    emit_event("done", "video", "Video composition completed.", metadata["steps"]["video"])

    metadata["total_seconds"] = round(time.time() - start, 3)
    metadata["artifacts"] = {
        "ocr_markdown": str(md_path),
        "ocr_assets": str(result_dir / "ocr_assets.json"),
        "plan_json": str(result_dir / "plan.json"),
        "slides_tex": str(tex_path),
        "slides_pdf": str(pdf_out),
        "script": str(script_path),
        "cursor": str(cursor_path),
        "subtitles": str(srt_path),
        "video": str(final_video),
    }
    write_json(result_dir / "sat.json", metadata)
    write_json(result_dir / "token.json", {"model": args.model, "mode": "ollama_api"})
    emit_event("done", "pipeline", "Pipeline completed.", metadata["artifacts"])
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real paper-to-video pipeline using MinerU, Ollama, Beamer, F5TTS, and ffmpeg.")
    parser.add_argument("--paper_pdf", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--goal_prompt", default="Create a rigorous academic video presentation from this paper.")
    parser.add_argument("--desired_minutes", type=int, default=6)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama_url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--mineru_method", default="ocr", choices=["ocr", "txt", "auto"])
    parser.add_argument("--ref_audio", default=str(ROOT / "assets" / "demo" / "reference.wav"))
    parser.add_argument("--ref_text", default=None)
    return parser.parse_args()


def main() -> None:
    metadata = run_pipeline(parse_args())
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


