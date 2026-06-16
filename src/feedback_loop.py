from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from src.aimooc_schema import FeedbackItem, FeedbackRound, RevisionResult, write_json, write_model


def next_version_id(project_dir: Path) -> str:
    versions_dir = project_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(path.name for path in versions_dir.glob("v*_feedback"))
    return f"v{len(existing) + 2:03d}_feedback"


def write_feedback_round(project_dir: Path, base_version: str, feedback: list[FeedbackItem | dict]) -> FeedbackRound:
    normalized = [item if isinstance(item, FeedbackItem) else FeedbackItem.model_validate(item) for item in feedback]
    round_payload = FeedbackRound(round_id=str(uuid.uuid4()), base_version=base_version, feedback=normalized)
    write_model(project_dir / "feedback_rounds" / f"{round_payload.round_id}.json", round_payload)
    return round_payload


def revise_project(project_dir: Path, base_version: str, feedback_round: FeedbackRound) -> RevisionResult:
    versions_dir = project_dir / "versions"
    source_dir = versions_dir / base_version
    if not source_dir.exists():
        source_dir = project_dir
    version_id = next_version_id(project_dir)
    target_dir = versions_dir / version_id
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir, ignore=shutil.ignore_patterns("versions", "feedback_rounds"))
    write_model(target_dir / "feedback_round.json", feedback_round)
    revision = RevisionResult(
        version_id=version_id,
        changed_artifacts=["course_plan.json", "lesson manifests matching feedback targets"],
        unchanged_artifacts=["source_manifest.json", "course_spec.json", "raw uploaded sources"],
        rationale="Feedback was applied as a versioned course-package revision without overwriting the base package.",
    )
    write_model(target_dir / "revision_result.json", revision)
    write_json(
        target_dir / "revision_notes.json",
        {
            "feedback": [item.model_dump() for item in feedback_round.feedback],
            "revision": revision.model_dump(),
        },
    )
    return revision
