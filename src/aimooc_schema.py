from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


SourceRole = Literal["primary", "reference", "prerequisite", "assignment", "reading"]
SourceType = Literal["pdf", "pptx", "docx", "md", "txt", "html", "url", "transcript", "unknown"]
AgenticFramework = Literal["langgraph", "hermes_adapter"]


class SourceItem(BaseModel):
    source_id: str
    filename: str
    path: str
    source_type: SourceType = "unknown"
    role: SourceRole = "reference"
    priority: int = Field(default=3, ge=1, le=5)
    title: str | None = None
    notes: str = ""


class SourceManifest(BaseModel):
    project_id: str
    sources: list[SourceItem]
    created_at: float = Field(default_factory=time.time)


class AIMOOCSpec(BaseModel):
    course_title: str = Field(min_length=2)
    audience: str = Field(min_length=2)
    learning_objectives: list[str] = Field(default_factory=list)
    requirements: str = ""
    total_minutes: int = Field(ge=3, le=600)
    module_count: int = Field(ge=1, le=20)
    lessons_per_module: int = Field(ge=1, le=20)
    preferred_style: str = "teaching_walkthrough"
    language: str = "zh-TW"
    difficulty: str = "intermediate"
    include_quizzes: bool = True
    include_assignments: bool = False
    include_avatar: bool = True
    avatar_mode: str = "presenter_card"
    feedback_mode: bool = True
    agentic_framework: AgenticFramework = "langgraph"


class LessonPlan(BaseModel):
    lesson_id: str
    module_id: str
    title: str
    objectives: list[str]
    source_ids: list[str]
    target_minutes: int
    artifacts: dict[str, str] = Field(default_factory=dict)


class ModulePlan(BaseModel):
    module_id: str
    title: str
    objectives: list[str]
    lessons: list[LessonPlan]


class CoursePlan(BaseModel):
    project_id: str
    course_title: str
    audience: str
    total_minutes: int
    language: str
    difficulty: str
    modules: list[ModulePlan]
    agentic_framework: AgenticFramework
    source_ids: list[str]
    created_at: float = Field(default_factory=time.time)


class FeedbackItem(BaseModel):
    target_type: str
    target_id: str
    severity: str = "normal"
    instruction: str
    preferred_action: str = "revise"


class FeedbackRound(BaseModel):
    round_id: str
    base_version: str
    feedback: list[FeedbackItem]
    created_at: float = Field(default_factory=time.time)


class RevisionResult(BaseModel):
    version_id: str
    changed_artifacts: list[str]
    unchanged_artifacts: list[str]
    rationale: str
    created_at: float = Field(default_factory=time.time)


class AvatarConfig(BaseModel):
    avatar_mode: str = "presenter_card"
    avatar_id: str = "default_teacher"
    position: str = "bottom_right"
    size: str = "medium"
    style: str = "academic"
    expression_policy: str = "calm"
    gesture_policy: str = "minimal"
    lip_sync: bool = False


class PackageManifest(BaseModel):
    project_id: str
    version_id: str
    source_manifest: str
    course_spec: str
    course_plan: str
    lessons: list[dict[str, Any]]
    feedback_rounds: list[str] = Field(default_factory=list)
    avatar_config: str | None = None
    created_at: float = Field(default_factory=time.time)


def write_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
