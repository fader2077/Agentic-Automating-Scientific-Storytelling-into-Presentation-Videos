from __future__ import annotations

import math
from pathlib import Path

from src.agentic_frameworks import run_agentic_trace
from src.aimooc_schema import AIMOOCSpec, CoursePlan, LessonPlan, ModulePlan, SourceManifest, write_json, write_model


def compact_objective_pool(spec: AIMOOCSpec) -> list[str]:
    objectives = [item.strip() for item in spec.learning_objectives if item.strip()]
    if objectives:
        return objectives
    return [
        f"Understand core ideas behind {spec.course_title}",
        "Connect source material to practical examples",
        "Explain methods, evidence, and limitations clearly",
    ]


def build_course_plan(project_id: str, manifest: SourceManifest, spec: AIMOOCSpec) -> CoursePlan:
    objectives = compact_objective_pool(spec)
    lesson_count = spec.module_count * spec.lessons_per_module
    minutes_per_lesson = max(3, math.floor(spec.total_minutes / max(lesson_count, 1)))
    source_ids = [source.source_id for source in manifest.sources]
    modules: list[ModulePlan] = []
    for module_idx in range(1, spec.module_count + 1):
        module_id = f"module_{module_idx:02d}"
        module_objective = objectives[(module_idx - 1) % len(objectives)]
        lessons: list[LessonPlan] = []
        for lesson_idx in range(1, spec.lessons_per_module + 1):
            global_idx = (module_idx - 1) * spec.lessons_per_module + lesson_idx
            lesson_id = f"{module_id}_lesson_{lesson_idx:02d}"
            lesson_source = source_ids[(global_idx - 1) % len(source_ids)] if source_ids else []
            lessons.append(
                LessonPlan(
                    lesson_id=lesson_id,
                    module_id=module_id,
                    title=f"Lesson {global_idx}: {module_objective}",
                    objectives=[
                        module_objective,
                        f"Apply the idea for {spec.audience}",
                        "Check understanding with a short formative task",
                    ],
                    source_ids=[lesson_source] if lesson_source else [],
                    target_minutes=minutes_per_lesson,
                )
            )
        modules.append(
            ModulePlan(
                module_id=module_id,
                title=f"Module {module_idx}: {module_objective}",
                objectives=[module_objective],
                lessons=lessons,
            )
        )
    return CoursePlan(
        project_id=project_id,
        course_title=spec.course_title,
        audience=spec.audience,
        total_minutes=spec.total_minutes,
        language=spec.language,
        difficulty=spec.difficulty,
        modules=modules,
        agentic_framework=spec.agentic_framework,
        source_ids=source_ids,
    )


def write_course_plan(result_dir: Path, manifest: SourceManifest, spec: AIMOOCSpec) -> CoursePlan:
    plan = build_course_plan(manifest.project_id, manifest, spec)
    write_model(result_dir / "course_plan.json", plan)
    trace = run_agentic_trace(
        spec.agentic_framework,
        manifest.project_id,
        ["SourceIngestionAgent", "CourseUnderstandingAgent", "CoursePlannerAgent", "ModulePlannerAgent", "LessonBuilderAgent"],
    )
    write_json(result_dir / "agentic_trace.json", trace)
    return plan
