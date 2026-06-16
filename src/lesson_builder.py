from __future__ import annotations

from pathlib import Path

from src.aimooc_schema import AIMOOCSpec, CoursePlan, LessonPlan, write_json


def lesson_slides(lesson: LessonPlan, spec: AIMOOCSpec) -> list[dict[str, object]]:
    return [
        {"title": lesson.title, "bullets": [lesson.objectives[0], f"Audience: {spec.audience}", f"Difficulty: {spec.difficulty}"]},
        {"title": "Core Concept", "bullets": ["Key idea from selected sources", "Why it matters", "Common misconception"]},
        {"title": "Worked Example", "bullets": ["Concrete teaching example", "Step-by-step reasoning", "Expected learner checkpoint"]},
        {"title": "Check Understanding", "bullets": ["One-minute recap", "Short formative quiz", "Next lesson bridge"]},
    ]


def lesson_script(lesson: LessonPlan, spec: AIMOOCSpec) -> list[dict[str, str]]:
    return [
        {"slide": "intro", "narration": f"This lesson introduces {lesson.title} for {spec.audience}."},
        {"slide": "core", "narration": f"We focus on {lesson.objectives[0]} and connect it to the uploaded course sources."},
        {"slide": "example", "narration": "A concrete example makes the concept easier to transfer into practice."},
        {"slide": "quiz", "narration": "The final checkpoint verifies whether the learner can explain the core idea."},
    ]


def lesson_quiz(lesson: LessonPlan) -> list[dict[str, object]]:
    return [
        {
            "question": f"What is the main learning goal of {lesson.title}?",
            "choices": lesson.objectives[:3],
            "answer": lesson.objectives[0],
        }
    ]


def build_lesson_package(course_plan: CoursePlan, spec: AIMOOCSpec, result_dir: Path) -> list[dict[str, object]]:
    lesson_records: list[dict[str, object]] = []
    for module in course_plan.modules:
        for lesson in module.lessons:
            lesson_dir = result_dir / lesson.lesson_id
            slides = lesson_slides(lesson, spec)
            script = lesson_script(lesson, spec)
            quiz = lesson_quiz(lesson) if spec.include_quizzes else []
            write_json(lesson_dir / "slides.json", {"lesson_id": lesson.lesson_id, "slides": slides})
            write_json(lesson_dir / "script.json", {"lesson_id": lesson.lesson_id, "script": script})
            write_json(lesson_dir / "quiz.json", {"lesson_id": lesson.lesson_id, "quiz": quiz})
            write_json(
                lesson_dir / "lesson_manifest.json",
                {
                    "lesson_id": lesson.lesson_id,
                    "module_id": lesson.module_id,
                    "title": lesson.title,
                    "target_minutes": lesson.target_minutes,
                    "artifacts": {
                        "slides": str(lesson_dir / "slides.json"),
                        "script": str(lesson_dir / "script.json"),
                        "quiz": str(lesson_dir / "quiz.json"),
                    },
                },
            )
            lesson_records.append(
                {
                    "lesson_id": lesson.lesson_id,
                    "module_id": lesson.module_id,
                    "title": lesson.title,
                    "dir": str(lesson_dir),
                    "artifacts": ["slides.json", "script.json", "quiz.json", "lesson_manifest.json"],
                }
            )
    return lesson_records
