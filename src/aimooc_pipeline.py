from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aimooc_schema import AIMOOCSpec, AvatarConfig, PackageManifest, SourceManifest, write_json, write_model
from src.avatar_director import build_avatar_config
from src.avatar_renderer import render_avatar_manifest
from src.course_planner import write_course_plan
from src.feedback_loop import revise_project, write_feedback_round
from src.lesson_builder import build_lesson_package
from src.source_ingest import validate_source_manifest


def load_model(path: Path, model_type):
    return model_type.model_validate(json.loads(path.read_text(encoding="utf-8-sig")))


def pdf_sources(manifest: SourceManifest) -> list[Path]:
    sources: list[Path] = []
    for source in sorted(manifest.sources, key=lambda item: (item.priority, item.role != "primary", item.filename)):
        path = Path(source.path)
        if source.source_type == "pdf" and path.exists():
            sources.append(path)
    return sources


def merge_pdf_sources(manifest: SourceManifest, output_pdf: Path) -> dict[str, object]:
    inputs = pdf_sources(manifest)
    if not inputs:
        raise ValueError("No readable PDF source found for AIMOOC video generation")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    import fitz

    merged = fitz.open()
    page_counts: list[dict[str, object]] = []
    for source_pdf in inputs:
        with fitz.open(source_pdf) as doc:
            merged.insert_pdf(doc)
            page_counts.append({"path": str(source_pdf), "pages": doc.page_count})
    merged.save(output_pdf)
    merged.close()
    return {"bundle_pdf": str(output_pdf), "sources": page_counts, "source_count": len(inputs)}


def course_video_prompt(spec: AIMOOCSpec, manifest: SourceManifest) -> str:
    source_lines = []
    for source in manifest.sources:
        source_lines.append(f"- {source.role} priority {source.priority}: {source.title or source.filename}")
    objectives = "; ".join(spec.learning_objectives) if spec.learning_objectives else "teach the central concepts"
    return (
        f"Create a multi-source teaching video for {spec.course_title}.\n"
        f"Audience: {spec.audience}.\n"
        f"Learning objectives: {objectives}.\n"
        f"Requirements: {spec.requirements or 'faithfully synthesize all primary and reference sources'}.\n"
        f"Use source roles and priorities:\n" + "\n".join(source_lines) + "\n"
        "Do not treat this as a single paper summary; synthesize a coherent lesson across sources."
    )


def run_course_video_pipeline(
    manifest: SourceManifest,
    spec: AIMOOCSpec,
    result_dir: Path,
    pipeline_python: Path | None = None,
    pipeline_script: Path | None = None,
    model: str = "qwen3.6:27b",
    ollama_url: str = "http://127.0.0.1:11434",
    temperature: float = 0.2,
    top_p: float = 0.9,
    avatar_image: Path | None = None,
) -> dict[str, object]:
    t0 = time.time()
    manifest_path = result_dir / "course_video_manifest.json"
    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached_video = Path(str(cached.get("video", "")))
        if cached_video.exists():
            cached["cached"] = True
            return cached
    bundle_pdf = result_dir / "source_bundle.pdf"
    bundle_meta = merge_pdf_sources(manifest, bundle_pdf)
    video_dir = result_dir / "generated_course_video"
    script = pipeline_script or (ROOT / "src" / "real_pipeline.py")
    python_exe = pipeline_python or Path(sys.executable)
    command = [
        str(python_exe),
        str(script),
        "--paper_pdf",
        str(bundle_pdf),
        "--result_dir",
        str(video_dir),
        "--desired_minutes",
        str(spec.total_minutes),
        "--goal_prompt",
        course_video_prompt(spec, manifest),
        "--model",
        model,
        "--ollama_url",
        ollama_url,
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--mineru_method",
        "ocr",
        "--narration_mode",
        "course",
    ]
    if spec.target_slide_count:
        command.extend(["--target_slides", str(spec.target_slide_count)])
    # Paper2Video-style separation: first render slides, speech, cursor, and
    # subtitles; AvatarRenderer then adds presenter/talking-head media.
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    run_log = result_dir / "course_video_pipeline.log"
    with run_log.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, text=True, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=7200)
    if completed.returncode != 0:
        raise RuntimeError(f"AIMOOC course video pipeline failed with exit code {completed.returncode}; see {run_log}")
    sat_path = video_dir / "sat.json"
    if not sat_path.exists():
        raise RuntimeError(f"AIMOOC course video pipeline did not write sat.json: {sat_path}")
    sat = json.loads(sat_path.read_text(encoding="utf-8"))
    artifacts = sat.get("artifacts", {})
    video_path = Path(str(artifacts.get("video", "")))
    if not video_path.exists():
        raise RuntimeError(f"AIMOOC course video artifact missing: {video_path}")
    payload = {
        "seconds": round(time.time() - t0, 3),
        "bundle": bundle_meta,
        "command": command,
        "run_log": str(run_log),
        "result_dir": str(video_dir),
        "sat": str(sat_path),
        "video": str(video_path),
        "slides_pdf": artifacts.get("slides_pdf", ""),
        "slides_tex": artifacts.get("slides_tex", ""),
        "subtitles": artifacts.get("subtitles", ""),
        "script": artifacts.get("script", ""),
        "duration": sat.get("steps", {}).get("video", {}).get("duration"),
        "slide_count": sat.get("steps", {}).get("beamer", {}).get("slide_count"),
    }
    write_json(result_dir / "course_video_manifest.json", payload)
    return payload


def run_aimooc_pipeline(
    source_manifest_path: Path,
    course_spec_path: Path,
    result_dir: Path,
    avatar_config_path: Path | None = None,
    feedback_round_path: Path | None = None,
    resume_from: Path | None = None,
    render_media: bool = False,
    lesson_video_source: Path | None = None,
    avatar_image: Path | None = None,
    generate_video_from_sources: bool = False,
    pipeline_python: Path | None = None,
    pipeline_script: Path | None = None,
    model: str = "qwen3.6:27b",
    ollama_url: str = "http://127.0.0.1:11434",
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> dict[str, object]:
    manifest = load_model(source_manifest_path, SourceManifest)
    spec = load_model(course_spec_path, AIMOOCSpec)
    issues = validate_source_manifest(manifest)
    if issues:
        raise ValueError("; ".join(issues))

    result_dir.mkdir(parents=True, exist_ok=True)
    write_model(result_dir / "source_manifest.json", manifest)
    write_model(result_dir / "course_spec.json", spec)
    plan = write_course_plan(result_dir, manifest, spec)
    lessons = build_lesson_package(plan, spec, result_dir)
    generated_course_video: dict[str, object] | None = None
    if render_media and lesson_video_source is None and generate_video_from_sources:
        generated_course_video = run_course_video_pipeline(
            manifest,
            spec,
            result_dir,
            pipeline_python=pipeline_python,
            pipeline_script=pipeline_script,
            model=model,
            ollama_url=ollama_url,
            temperature=temperature,
            top_p=top_p,
            avatar_image=avatar_image,
        )
        lesson_video_source = Path(str(generated_course_video["video"]))

    if avatar_config_path and avatar_config_path.exists():
        avatar_config = load_model(avatar_config_path, AvatarConfig)
    else:
        avatar_config = build_avatar_config(spec)
    write_model(result_dir / "avatar_config.json", avatar_config)
    for lesson in lessons:
        lesson_dir = Path(str(lesson["dir"]))
        artifacts = list(lesson.get("artifacts", []))
        source_video_for_lesson: Path | None = None
        if lesson_video_source and lesson_video_source.exists():
            source_video_for_lesson = lesson_dir / "video.mp4"
            shutil.copyfile(lesson_video_source, source_video_for_lesson)
            if "video.mp4" not in artifacts:
                artifacts.append("video.mp4")
        avatar_manifest = render_avatar_manifest(
            lesson_dir,
            avatar_config,
            render_media=render_media or bool(source_video_for_lesson),
            source_video=source_video_for_lesson,
            avatar_image=avatar_image,
        )
        if avatar_manifest.get("video") and "avatar_video.mp4" not in artifacts:
            artifacts.append("avatar_video.mp4")
        if "avatar_manifest.json" not in artifacts:
            artifacts.append("avatar_manifest.json")
        lesson["artifacts"] = artifacts
        lesson_manifest_path = lesson_dir / "lesson_manifest.json"
        if lesson_manifest_path.exists():
            lesson_manifest = json.loads(lesson_manifest_path.read_text(encoding="utf-8"))
            lesson_manifest.setdefault("artifacts", {})
            if source_video_for_lesson:
                lesson_manifest["artifacts"]["video"] = str(source_video_for_lesson)
            if avatar_manifest.get("video"):
                lesson_manifest["artifacts"]["avatar_video"] = str(avatar_manifest["video"])
            lesson_manifest["artifacts"]["avatar_manifest"] = str(lesson_dir / "avatar_manifest.json")
            write_json(lesson_manifest_path, lesson_manifest)

    feedback_paths: list[str] = []
    if feedback_round_path and feedback_round_path.exists():
        round_payload = json.loads(feedback_round_path.read_text(encoding="utf-8"))
        feedback_round = write_feedback_round(result_dir, round_payload.get("base_version", "v001_initial"), round_payload.get("feedback", []))
        revision = revise_project(result_dir, feedback_round.base_version, feedback_round)
        feedback_paths.append(str(result_dir / "feedback_rounds" / f"{feedback_round.round_id}.json"))
        write_model(result_dir / "revision_result.json", revision)

    package = PackageManifest(
        project_id=manifest.project_id,
        version_id="v001_initial",
        source_manifest=str(result_dir / "source_manifest.json"),
        course_spec=str(result_dir / "course_spec.json"),
        course_plan=str(result_dir / "course_plan.json"),
        lessons=lessons,
        feedback_rounds=feedback_paths,
        avatar_config=str(result_dir / "avatar_config.json"),
        course_video=generated_course_video,
    )
    write_model(result_dir / "course_package_manifest.json", package)
    payload = package.model_dump()

    versions_dir = result_dir / "versions"
    initial_dir = versions_dir / "v001_initial"
    if initial_dir.exists():
        shutil.rmtree(initial_dir)
    shutil.copytree(
        result_dir,
        initial_dir,
        ignore=shutil.ignore_patterns("versions", "generated_course_video", "source_bundle.pdf"),
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--course_spec", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--avatar_config", default="")
    parser.add_argument("--feedback_round", default="")
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--render_media", action="store_true")
    parser.add_argument("--lesson_video_source", default="")
    parser.add_argument("--avatar_image", default="")
    parser.add_argument("--generate_video_from_sources", action="store_true")
    parser.add_argument("--pipeline_python", default="")
    parser.add_argument("--pipeline_script", default="")
    parser.add_argument("--model", default="qwen3.6:27b")
    parser.add_argument("--ollama_url", default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_aimooc_pipeline(
        Path(args.source_manifest),
        Path(args.course_spec),
        Path(args.result_dir),
        Path(args.avatar_config) if args.avatar_config else None,
        Path(args.feedback_round) if args.feedback_round else None,
        Path(args.resume_from) if args.resume_from else None,
        render_media=bool(args.render_media),
        lesson_video_source=Path(args.lesson_video_source) if args.lesson_video_source else None,
        avatar_image=Path(args.avatar_image) if args.avatar_image else None,
        generate_video_from_sources=bool(args.generate_video_from_sources),
        pipeline_python=Path(args.pipeline_python) if args.pipeline_python else None,
        pipeline_script=Path(args.pipeline_script) if args.pipeline_script else None,
        model=args.model,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
