from __future__ import annotations

import argparse
import json
import shutil
import sys
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

    versions_dir = result_dir / "versions"
    initial_dir = versions_dir / "v001_initial"
    if initial_dir.exists():
        shutil.rmtree(initial_dir)
    shutil.copytree(result_dir, initial_dir, ignore=shutil.ignore_patterns("versions"))

    package = PackageManifest(
        project_id=manifest.project_id,
        version_id="v001_initial",
        source_manifest=str(result_dir / "source_manifest.json"),
        course_spec=str(result_dir / "course_spec.json"),
        course_plan=str(result_dir / "course_plan.json"),
        lessons=lessons,
        feedback_rounds=feedback_paths,
        avatar_config=str(result_dir / "avatar_config.json"),
    )
    write_model(result_dir / "course_package_manifest.json", package)
    return package.model_dump()


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
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
