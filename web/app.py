from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agentic_graph import agentic_graph_status
from src.agentic_frameworks import available_frameworks, normalize_framework
from src.aimooc_pipeline import run_aimooc_pipeline
from src.aimooc_schema import AIMOOCSpec, FeedbackItem, SourceItem, SourceManifest, write_model
from src.feedback_loop import revise_project, write_feedback_round
from src.source_ingest import build_source_item, build_source_manifest


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
TASK_DIR = DATA_DIR / "tasks"
PROJECT_DB = DATA_DIR / "aimooc.sqlite3"
SETTINGS_PATH = DATA_DIR / "settings.json"
STATIC_DIR = ROOT / "static"
AIMOOC_STATIC_DIR = STATIC_DIR / "aimooc"
SAMPLE_VIDEO_CANDIDATES = []
ARTIFACT_FILES = {}
LOCAL_OLLAMA_HOSTS = {"localhost", "127.0.0.1", "::1"}
PIPELINE_STEP_MAP = {
    "mineru_ocr": "ingest",
    "ollama_plan": "planner",
    "beamer": "slides",
    "script": "script",
    "tts": "tts",
    "cursor": "cursor",
    "video": "compose",
}
PIPELINE_PYTHON_CANDIDATES = [
    ROOT.parent / ".runtime_env" / "Scripts" / "python.exe",
    ROOT.parent / ".venv" / "Scripts" / "python.exe",
]
PIPELINE_SCRIPT = ROOT.parent / "src" / "real_pipeline.py"
AGENTS_DIR = ROOT.parent / "src" / "agents"
AGENTS_MANIFEST = AGENTS_DIR / "manifest.json"
TOOLS_MANIFEST = ROOT.parent / "src" / "tools" / "manifest.json"
WEB_RESULT_DIR = ROOT.parent / "result" / "web_jobs"
AIMOOC_RESULT_DIR = ROOT.parent / "result" / "aimooc_projects"

for directory in [DATA_DIR, UPLOAD_DIR, TASK_DIR, STATIC_DIR, AIMOOC_STATIC_DIR, WEB_RESULT_DIR, AIMOOC_RESULT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_STEP_TICKS = {
    "ingest": 16,
    "planner": 18,
    "slides": 22,
    "script": 18,
    "tts": 20,
    "cursor": 16,
    "compose": 18,
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "ollama_url": "http://127.0.0.1:11434",
    "text_model": "qwen3.6:27b",
    "vision_model": "qwen2.5vl:latest",
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 4096,
    "system_prompt": (
        "You coordinate a presentation-video agent workflow. Preserve academic fidelity, "
        "plan slide structure carefully, and keep narration concise."
    ),
    "tick_seconds": 1,
    "step_ticks": DEFAULT_STEP_TICKS,
}

WORKER_LOCK = threading.Lock()
TASK_IO_LOCK = threading.Lock()
QUEUE_EVENT = threading.Event()
WORKER_THREAD: threading.Thread | None = None
WORKER_ID = "control-room-worker-1"
SKILL_EDIT_LOCK = threading.Lock()

SLIDE_STYLE_TEMPLATES: list[dict[str, Any]] = [
    {
        "key": "clean_academic",
        "title": "Clean Academic",
        "value": "clean academic beamer with figure-led slides",
        "accent": "#126b9a",
        "preview": {
            "layout": "two-column",
            "headline": "Problem, method, evidence",
            "visual": "large OCR figure with compact bullets",
            "notes": "Balanced overview deck for most papers.",
        },
    },
    {
        "key": "dense_methods",
        "title": "Dense Methods",
        "value": "dense methods-first beamer with equations, algorithms, and compact comparison tables",
        "accent": "#6a4c93",
        "preview": {
            "layout": "equation-led",
            "headline": "Model and objective first",
            "visual": "OCR formulas and algorithm blocks",
            "notes": "Best for theory, model architecture, and derivations.",
        },
    },
    {
        "key": "visual_results",
        "title": "Visual Results",
        "value": "visual results deck with large OCR figures, benchmark tables, and concise takeaways",
        "accent": "#2f7d32",
        "preview": {
            "layout": "visual-first",
            "headline": "Experiments and ablations",
            "visual": "wide tables, figures, and charts",
            "notes": "Best for empirical papers with many result assets.",
        },
    },
    {
        "key": "teaching_walkthrough",
        "title": "Teaching Walkthrough",
        "value": "teaching walkthrough beamer with progressive motivation, method intuition, and evaluation summary",
        "accent": "#a45113",
        "preview": {
            "layout": "progressive",
            "headline": "Motivation to conclusion",
            "visual": "small diagrams plus short explanations",
            "notes": "Best for talks aimed at mixed technical audiences.",
        },
    },
]

class SettingsPayload(BaseModel):
    ollama_url: str = Field(min_length=1)
    text_model: str = Field(min_length=1)
    vision_model: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(ge=0.0, le=1.0)
    max_tokens: int = Field(ge=256, le=32768)
    system_prompt: str = Field(min_length=1)
    tick_seconds: int = Field(ge=1, le=10)
    step_ticks: dict[str, int]


class CreateTaskPayload(BaseModel):
    upload_id: str
    goal_prompt: str = Field(min_length=10)
    desired_minutes: int = Field(ge=1, le=20)
    target_slide_count: int = Field(default=12, ge=5, le=30)
    preferred_slide_style: str = Field(min_length=2)
    agentic_framework: str = "langgraph"
    avatar_mode: str = "presenter_card"
    avatar_position: str = "bottom_right"


class AIMOOCSourceSelection(BaseModel):
    source_id: str
    role: str = "reference"
    priority: int = Field(default=3, ge=1, le=5)
    title: str | None = None
    notes: str = ""


class CreateAIMOOCProjectPayload(BaseModel):
    source_ids: list[str] = Field(default_factory=list)
    sources: list[AIMOOCSourceSelection] = Field(default_factory=list)
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
    agentic_framework: str = "langgraph"
    render_videos: bool = False
    lesson_video_task_id: str | None = None


class AIMOOCFeedbackPayload(BaseModel):
    base_version: str = "v001_initial"
    feedback: list[FeedbackItem]


class OllamaTestPayload(BaseModel):
    ollama_url: str = Field(min_length=1)
    text_model: str = Field(min_length=1)
    vision_model: str = Field(min_length=1)

class SkillsUpdatePayload(BaseModel):
    content: str = Field(min_length=1, max_length=20000)


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(PROJECT_DB)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS aimooc_projects (
            project_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            framework TEXT NOT NULL,
            result_dir TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS aimooc_versions (
            project_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            path TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY(project_id, version_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS aimooc_feedback (
            project_id TEXT NOT NULL,
            round_id TEXT NOT NULL,
            base_version TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY(project_id, round_id)
        )
        """
    )
    conn.commit()
    return conn


def upsert_project(project_id: str, title: str, status: str, framework: str, result_dir: Path, payload: dict[str, Any]) -> None:
    now = now_ts()
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO aimooc_projects(project_id, title, status, framework, result_dir, created_at, updated_at, payload_json)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
              title=excluded.title,
              status=excluded.status,
              framework=excluded.framework,
              result_dir=excluded.result_dir,
              updated_at=excluded.updated_at,
              payload_json=excluded.payload_json
            """,
            (project_id, title, status, framework, str(result_dir), now, now, json.dumps(payload, ensure_ascii=False)),
        )
        conn.commit()


def add_project_version(project_id: str, version_id: str, path: Path) -> None:
    with db_connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO aimooc_versions(project_id, version_id, path, created_at)
            VALUES(?, ?, ?, ?)
            """,
            (project_id, version_id, str(path), now_ts()),
        )
        conn.commit()


def project_row(project_id: str) -> sqlite3.Row:
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM aimooc_projects WHERE project_id=?", (project_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="AIMOOC project not found")
    return row


def project_summary(row: sqlite3.Row) -> dict[str, Any]:
    payload = json.loads(row["payload_json"])
    return {
        "project_id": row["project_id"],
        "title": row["title"],
        "status": row["status"],
        "framework": row["framework"],
        "result_dir": row["result_dir"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "source_count": len(payload.get("source_manifest", {}).get("sources", [])),
        "version_id": payload.get("package", {}).get("version_id", "v001_initial"),
    }

def normalize_step_ticks(raw: dict[str, Any] | None) -> dict[str, int]:
    merged = dict(DEFAULT_STEP_TICKS)
    if raw:
        for key, default_value in DEFAULT_STEP_TICKS.items():
            value = raw.get(key, default_value)
            merged[key] = max(10, min(60, int(value)))
    return merged


def normalize_ollama_url(raw_url: str) -> str:
    url = raw_url.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=422, detail="Ollama URL must be a valid local HTTP endpoint.")
    if parsed.hostname not in LOCAL_OLLAMA_HOSTS:
        raise HTTPException(status_code=422, detail="Ollama URL must point to the local host.")
    return url


def resolve_sample_video() -> Path | None:
    for candidate in SAMPLE_VIDEO_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def resolve_artifact(name: str) -> Path:
    path = ARTIFACT_FILES.get(name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return path


def resolve_task_artifact(task_id: str, artifact_name: str) -> Path:
    task = read_task(task_id)
    artifact_paths = task.get("artifact_paths", {})
    raw_path = artifact_paths.get(artifact_name)
    if not raw_path:
        raise HTTPException(status_code=404, detail="Artifact not found")
    path = Path(raw_path).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact missing on disk")
    return path


def load_settings() -> dict[str, Any]:
    if SETTINGS_PATH.exists():
        raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    else:
        raw = dict(DEFAULT_SETTINGS)
    settings = dict(DEFAULT_SETTINGS)
    settings.update(raw)
    settings["ollama_url"] = normalize_ollama_url(settings["ollama_url"])
    settings["tick_seconds"] = max(1, min(10, int(settings.get("tick_seconds", 1))))
    settings["step_ticks"] = normalize_step_ticks(settings.get("step_ticks"))
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return settings


def save_settings(payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_SETTINGS)
    merged.update(payload)
    merged["ollama_url"] = normalize_ollama_url(str(merged["ollama_url"]))
    merged["tick_seconds"] = max(1, min(10, int(merged.get("tick_seconds", 1))))
    merged["step_ticks"] = normalize_step_ticks(merged.get("step_ticks"))
    SETTINGS_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def task_path(task_id: str) -> Path:
    return TASK_DIR / f"{task_id}.json"


def read_json_file(path: Path) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(5):
        try:
            with TASK_IO_LOCK:
                raw = path.read_text(encoding="utf-8")
            return json.loads(raw)
        except (JSONDecodeError, OSError) as exc:
            last_error = exc
            time.sleep(0.03)
    raise RuntimeError(f"Could not read stable JSON from {path}: {last_error}")


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    raw = json.dumps(payload, indent=2)
    with TASK_IO_LOCK:
        temp_path.write_text(raw, encoding="utf-8")
        os.replace(temp_path, path)


def read_task(task_id: str) -> dict[str, Any]:
    path = task_path(task_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    return read_json_file(path)


def write_task(task: dict[str, Any]) -> None:
    write_json_file(task_path(task["id"]), task)


def all_tasks() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for task_file in TASK_DIR.glob("*.json"):
        items.append(read_json_file(task_file))
    return items


def probe_ollama(ollama_url: str) -> dict[str, Any]:
    normalized = normalize_ollama_url(ollama_url)
    endpoint = f"{normalized}/api/tags"
    try:
        with urlopen(endpoint, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = [item.get("name", "") for item in payload.get("models", []) if item.get("name")]
        return {
            "ok": True,
            "reachable": True,
            "mode": "local",
            "message": f"Local Ollama endpoint responded at {normalized}.",
            "models": models[:12],
        }
    except (URLError, HTTPError, TimeoutError):
        return {
            "ok": True,
            "reachable": False,
            "mode": "local",
            "message": f"Local Ollama endpoint configured at {normalized}; live probe did not respond.",
            "models": [],
        }


def gpu_status() -> dict[str, Any]:
    try:
        import torch

        return {
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        return {"cuda_available": False, "error": str(exc)}


def resolve_pipeline_python() -> Path:
    env_python = os.environ.get("PIPELINE_RUNTIME_PYTHON")
    candidates = [Path(env_python)] if env_python else []
    candidates.extend(PIPELINE_PYTHON_CANDIDATES)
    candidates.append(Path(sys.executable))
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return Path(sys.executable)


def build_tool(
    key: str,
    title: str,
    purpose: str,
    calls: list[dict[str, str]],
) -> dict[str, Any]:
    registry_item = tool_map().get(key, {})
    return {
        "key": key,
        "title": registry_item.get("title", title),
        "purpose": registry_item.get("purpose", purpose),
        "kind": registry_item.get("kind", ""),
        "entrypoint": registry_item.get("entrypoint", ""),
        "uses_gpu": bool(registry_item.get("uses_gpu", False)),
        "status": "pending",
        "progress": 0.0,
        "tick_progress": 0,
        "tick_total": max(len(calls), 1),
        "calls": calls,
        "recent_call": "",
    }


def read_registry_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise HTTPException(status_code=500, detail=f"Registry must be a list: {path}")
    return payload


def tool_catalog() -> list[dict[str, Any]]:
    tools = read_registry_json(TOOLS_MANIFEST)
    return [
        {
            **tool,
            "manifest_path": str(TOOLS_MANIFEST),
        }
        for tool in tools
    ]


def tool_map() -> dict[str, dict[str, Any]]:
    return {tool["key"]: tool for tool in tool_catalog()}


def agent_catalog() -> list[dict[str, Any]]:
    tools_by_key = tool_map()
    agents = read_registry_json(AGENTS_MANIFEST)
    normalized: list[dict[str, Any]] = []
    for agent in agents:
        skills_md_rel = Path(str(agent.get("skills_md", "")))
        skills_md_path = (AGENTS_DIR / skills_md_rel).resolve()
        tool_keys = [str(key) for key in agent.get("tools", [])]
        normalized.append(
            {
                **agent,
                "tools": [tools_by_key.get(key, {"key": key, "title": key}) for key in tool_keys],
                "tool_keys": tool_keys,
                "skills_md_path": str(skills_md_path),
                "skills_md_url": f"/api/agents/{agent['key']}/skills.md",
                "manifest_path": str(AGENTS_MANIFEST),
            }
        )
    return normalized


def resolve_agent_skills_md(agent_key: str) -> Path:
    for agent in read_registry_json(AGENTS_MANIFEST):
        if agent.get("key") == agent_key:
            path = (AGENTS_DIR / str(agent.get("skills_md", ""))).resolve()
            if not str(path).startswith(str(AGENTS_DIR.resolve())) or not path.exists():
                raise HTTPException(status_code=404, detail="Agent skills file not found")
            return path
    raise HTTPException(status_code=404, detail="Agent not found")


def estimate_content_slide_count(desired_minutes: int, target_slide_count: int | None = None) -> int:
    if target_slide_count:
        return int(target_slide_count)
    minutes = max(1, int(desired_minutes))
    if minutes <= 3:
        return max(8, minutes * 3)
    if minutes <= 6:
        return max(10, min(14, minutes * 2 + 2))
    return max(14, min(18, minutes * 2))


def build_steps(payload: CreateTaskPayload, settings: dict[str, Any]) -> list[dict[str, Any]]:
    step_ticks = settings["step_ticks"]
    target_slides = estimate_content_slide_count(payload.desired_minutes, payload.target_slide_count)
    return [
        {
            "key": "ingest",
            "title": "Paper intake and OCR scan",
            "agent": "IngestionAgent",
            "detail": f"Collect document structure, figures, and references from the uploaded source. Final slide target: {target_slides}.",
            "status": "pending",
            "tick_total": step_ticks["ingest"],
            "tick_progress": 0,
            "progress": 0.0,
            "tools": [
                build_tool(
                    "pdf_manifest",
                    "PDF Manifest Reader",
                    "Read file metadata and page structure.",
                    [
                        {"name": "read_upload_manifest", "input": payload.upload_id, "output": "document inventory"},
                        {"name": "index_pages", "input": "paper pages", "output": "page map"},
                    ],
                ),
                build_tool(
                    "ocr_router",
                    "OCR Router",
                    "Extract text blocks and region candidates.",
                    [
                        {"name": "render_pdf_pages", "input": "pdf pages", "output": "page rasters"},
                        {"name": "scan_text_regions", "input": "page rasters", "output": "ocr spans"},
                        {"name": "collect_figure_boxes", "input": "page rasters", "output": "figure regions"},
                    ],
                ),
                build_tool(
                    "ocr_asset_manifest",
                    "OCR Asset Manifest",
                    "Normalize extracted OCR visual assets for slides and inspection.",
                    [
                        {"name": "read_content_list", "input": "MinerU content JSON", "output": "structured OCR blocks"},
                        {"name": "copy_visual_assets", "input": "images, tables, charts, equations", "output": "latex asset directory"},
                        {"name": "write_manifest", "input": "normalized OCR blocks", "output": "ocr_assets.json"},
                    ],
                ),
            ],
        },
        {
            "key": "planner",
            "title": "Multi-agent planning",
            "agent": "PlannerAgent",
            "detail": "Break the paper into a talk outline, pacing plan, and agent handoff graph.",
            "status": "pending",
            "tick_total": step_ticks["planner"],
            "tick_progress": 0,
            "progress": 0.0,
            "tools": [
                build_tool(
                    "section_planner",
                    "Section Planner",
                    "Allocate sections and slide budgets.",
                    [
                        {"name": "score_sections", "input": "ocr spans", "output": "section weights"},
                        {"name": "assign_slide_budget", "input": f"{payload.desired_minutes} minute target + {target_slides} selected slides", "output": f"{target_slides} final pages"},
                    ],
                ),
                build_tool(
                    "ollama_dispatch",
                    "Ollama Dispatch",
                    "Represent local model prompts through the local endpoint.",
                    [
                        {"name": "draft_agent_brief", "input": settings["text_model"], "output": "planner brief"},
                        {"name": "review_visual_needs", "input": settings["vision_model"], "output": "visual checklist"},
                    ],
                ),
            ],
        },
        {
            "key": "slides",
            "title": "Beamer slide drafting",
            "agent": "SlideBuilderAgent",
            "detail": f"Draft a {payload.preferred_slide_style} deck with figure-led layouts and paper-faithful structure.",
            "status": "pending",
            "tick_total": step_ticks["slides"],
            "tick_progress": 0,
            "progress": 0.0,
            "tools": [
                build_tool(
                    "beamer_writer",
                    "Beamer Writer",
                    "Generate slide frames and section ordering.",
                    [
                        {"name": "write_outline_frames", "input": "planner brief", "output": "frame scaffold"},
                        {"name": "place_core_claims", "input": "section weights", "output": "claim blocks"},
                        {"name": "compose_frame_notes", "input": "slide scaffold", "output": "speaker anchors"},
                    ],
                ),
                build_tool(
                    "figure_grounder",
                    "Figure Grounder",
                    "Place figures and tables into frame slots.",
                    [
                        {"name": "rank_figures", "input": "figure regions", "output": "ranked visuals"},
                        {"name": "bind_figure_slots", "input": "frame scaffold", "output": "visual placements"},
                    ],
                ),
            ],
        },
        {
            "key": "script",
            "title": "Narration and subtitle drafting",
            "agent": "ScriptAgent",
            "detail": "Prepare slide-level speaker notes, subtitle segments, and pacing checkpoints.",
            "status": "pending",
            "tick_total": step_ticks["script"],
            "tick_progress": 0,
            "progress": 0.0,
            "tools": [
                build_tool(
                    "script_planner",
                    "Script Planner",
                    "Draft narration and subtitle structure.",
                    [
                        {"name": "draft_slide_notes", "input": "frame scaffold", "output": "speaker notes"},
                        {"name": "segment_subtitles", "input": "speaker notes", "output": "subtitle spans"},
                        {"name": "mark_focus_beats", "input": "speaker notes", "output": "cursor anchors"},
                    ],
                ),
                build_tool(
                    "subtitle_aligner",
                    "Subtitle Aligner",
                    "Check chunking and narration rhythm before speech staging.",
                    [
                        {"name": "split_long_sentences", "input": "speaker notes", "output": "readable chunks"},
                        {"name": "assign_slide_windows", "input": "readable chunks", "output": "subtitle timeline"},
                    ],
                ),
            ],
        },
        {
            "key": "tts",
            "title": "TTS and voice conditioning",
            "agent": "SpeechAgent",
            "detail": "Stage speech batches, reference alignment, and narration packaging.",
            "status": "pending",
            "tick_total": step_ticks["tts"],
            "tick_progress": 0,
            "progress": 0.0,
            "tools": [
                build_tool(
                    "voice_profiler",
                    "Voice Profiler",
                    "Prepare reference voice metadata and alignment inputs.",
                    [
                        {"name": "inspect_reference_audio", "input": "reference wav", "output": "voice metadata"},
                        {"name": "prepare_alignment_text", "input": "subtitle timeline", "output": "voice prompt pack"},
                    ],
                ),
                build_tool(
                    "f5_queue",
                    "F5 Queue",
                    "Prepare local speech jobs and reference voice alignment.",
                    [
                        {"name": "register_reference_voice", "input": "reference wav", "output": "voice profile"},
                        {"name": "queue_slide_batches", "input": "subtitle spans", "output": "tts jobs"},
                        {"name": "finalize_audio_pack", "input": "tts jobs", "output": "audio bundle"},
                    ],
                ),
            ],
        },
        {
            "key": "cursor",
            "title": "Cursor grounding",
            "agent": "GroundingAgent",
            "detail": "Bind visual focus regions to narration beats and slide windows.",
            "status": "pending",
            "tick_total": step_ticks["cursor"],
            "tick_progress": 0,
            "progress": 0.0,
            "tools": [
                build_tool(
                    "layout_inspector",
                    "Layout Inspector",
                    "Read slide regions before cursor target routing.",
                    [
                        {"name": "read_slide_regions", "input": "slide deck", "output": "region map"},
                        {"name": "rank_visual_targets", "input": "region map", "output": "focus candidates"},
                    ],
                ),
                build_tool(
                    "cursor_router",
                    "Cursor Router",
                    "Convert focus beats into cursor trajectory plans.",
                    [
                        {"name": "load_focus_beats", "input": "subtitle anchors", "output": "focus timeline"},
                        {"name": "resolve_pointer_targets", "input": "figure placements", "output": "cursor targets"},
                        {"name": "write_cursor_track", "input": "cursor targets", "output": "cursor json"},
                    ],
                ),
            ],
        },
        {
            "key": "compose",
            "title": "Video composition",
            "agent": "RenderAgent",
            "detail": "Package slides, cursor layer, subtitles, and final MP4 handoff.",
            "status": "pending",
            "tick_total": step_ticks["compose"],
            "tick_progress": 0,
            "progress": 0.0,
            "tools": [
                build_tool(
                    "slide_renderer",
                    "Slide Renderer",
                    "Prepare slide image frames before video packaging.",
                    [
                        {"name": "render_pdf_frames", "input": "slides pdf", "output": "slide images"},
                        {"name": "verify_frame_size", "input": "slide images", "output": "frame manifest"},
                    ],
                ),
                build_tool(
                    "ffmpeg_packager",
                    "FFmpeg Packager",
                    "Assemble output assets into the final handoff video.",
                    [
                        {"name": "render_slide_track", "input": "slide deck", "output": "video track"},
                        {"name": "burn_subtitles", "input": "subtitle spans", "output": "captioned video"},
                        {"name": "mux_final_output", "input": "audio bundle + cursor json", "output": "final mp4"},
                    ],
                ),
                build_tool(
                    "mp4_verifier",
                    "MP4 Verifier",
                    "Verify final MP4 duration and output availability.",
                    [
                        {"name": "ffprobe_duration", "input": "final mp4", "output": "duration metadata"},
                    ],
                ),
            ],
        },
    ]


def append_log(task: dict[str, Any], message: str) -> None:
    task["log"].append(f"{time.strftime('%H:%M:%S')} {message}")


def now_ts() -> float:
    return time.time()


def task_counts(task: dict[str, Any]) -> dict[str, int]:
    return task.get("progress_counts", {"completed": 0, "total": 0})


def snapshot_task_state(task: dict[str, Any], label: str, step_key: str | None = None, tool_key: str | None = None) -> None:
    task["timeline"].append(
        {
            "index": len(task["timeline"]),
            "timestamp": now_ts(),
            "label": label,
            "step_key": step_key,
            "tool_key": tool_key,
            "status": task["status"],
            "current_step": task.get("current_step", -1),
            "progress": task.get("progress", 0.0),
            "progress_counts": task_counts(task),
            "steps": [
                {
                    "key": step["key"],
                    "title": step["title"],
                    "status": step["status"],
                    "progress": step.get("progress", 0.0),
                    "tick_progress": step.get("tick_progress", 0),
                    "tick_total": step.get("tick_total", 0),
                    "tools": [
                        {
                            "key": tool["key"],
                            "title": tool["title"],
                            "status": tool["status"],
                            "progress": tool.get("progress", 0.0),
                            "tick_progress": tool.get("tick_progress", 0),
                            "tick_total": tool.get("tick_total", 0),
                            "recent_call": tool.get("recent_call", ""),
                        }
                        for tool in step["tools"]
                    ],
                }
                for step in task["steps"]
            ],
        }
    )


def update_tool_progress(tool: dict[str, Any], current_tick: int, total_ticks: int) -> None:
    tool["tick_total"] = max(total_ticks, 1)
    tool["tick_progress"] = current_tick
    tool["progress"] = round(current_tick / max(total_ticks, 1), 3)


def queue_sort_key(task: dict[str, Any]) -> tuple[float, str]:
    queued_at = task.get("job", {}).get("queued_at", task.get("created_at", 0.0))
    return float(queued_at), task["id"]


def queued_tasks() -> list[dict[str, Any]]:
    items = [task for task in all_tasks() if task.get("job", {}).get("state") == "queued"]
    return sorted(items, key=queue_sort_key)


def running_task() -> dict[str, Any] | None:
    for task in all_tasks():
        if task.get("job", {}).get("state") == "running":
            return task
    return None


def refresh_queue_positions() -> None:
    queued = queued_tasks()
    for index, task in enumerate(queued, start=1):
        task["job"]["queue_position"] = index
        write_task(task)


def task_summary(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task["id"],
        "status": task["status"],
        "created_at": task["created_at"],
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "goal_prompt": task["goal_prompt"],
        "desired_minutes": task["desired_minutes"],
        "target_slide_count": task.get("target_slide_count"),
        "upload_name": task["upload_name"],
        "progress": task.get("progress", 0.0),
        "progress_counts": task_counts(task),
        "job": task.get("job", {}),
        "timeline_length": len(task.get("timeline", [])),
    }


def queue_state() -> dict[str, Any]:
    active = running_task()
    waiting = queued_tasks()
    return {
        "worker_id": WORKER_ID,
        "active_task_id": active["id"] if active else None,
        "queued_count": len(waiting),
        "queued": [
            {
                "id": task["id"],
                "upload_name": task["upload_name"],
                "queue_position": task["job"]["queue_position"],
                "created_at": task["created_at"],
                "desired_minutes": task["desired_minutes"],
                "target_slide_count": task.get("target_slide_count"),
            }
            for task in waiting
        ],
    }


def initialize_task_runtime(task: dict[str, Any]) -> None:
    task["status"] = "queued"
    task["progress"] = 0.0
    task["progress_counts"] = {
        "completed": 0,
        "total": sum(step["tick_total"] for step in task["steps"]),
    }
    task["current_step"] = -1
    task["timeline"] = []
    task["sample_video_url"] = None
    task["artifacts"] = {}
    task["started_at"] = None
    task["finished_at"] = None
    task["job"] = {
        "id": f"job-{task['id']}",
        "state": "queued",
        "worker_id": None,
        "attempt": 0,
        "queued_at": now_ts(),
        "started_at": None,
        "finished_at": None,
        "queue_position": 0,
        "last_heartbeat": None,
    }
    task["log"] = [
        f"{time.strftime('%H:%M:%S')} Task queued.",
        f"{time.strftime('%H:%M:%S')} Uploaded source: {task['upload_name']}.",
    ]
    snapshot_task_state(task, "Task queued.")


def find_step(task: dict[str, Any], step_key: str) -> dict[str, Any] | None:
    for step in task["steps"]:
        if step["key"] == step_key:
            return step
    return None


def mark_step_started(task: dict[str, Any], step_key: str, message: str) -> None:
    step = find_step(task, step_key)
    if not step:
        return
    task["current_step"] = task["steps"].index(step)
    step["status"] = "running"
    step["started_at"] = now_ts()
    for tool in step["tools"]:
        if tool["status"] == "pending":
            tool["status"] = "running"
            tool["recent_call"] = message
    append_log(task, message)
    snapshot_task_state(task, message, step_key)


def mark_step_completed(task: dict[str, Any], step_key: str, message: str, data: dict[str, Any]) -> None:
    step = find_step(task, step_key)
    if not step:
        return
    step["status"] = "completed"
    step["finished_at"] = now_ts()
    step["progress"] = 1.0
    step["tick_progress"] = step["tick_total"]
    for tool in step["tools"]:
        tool["status"] = "completed"
        tool["progress"] = 1.0
        tool["tick_progress"] = tool["tick_total"]
        tool["recent_call"] = message
    completed = sum(item["tick_total"] for item in task["steps"] if item["status"] == "completed")
    total = sum(item["tick_total"] for item in task["steps"])
    task["progress_counts"] = {"completed": completed, "total": total}
    task["progress"] = round(completed / max(total, 1), 3)
    append_log(task, f"{message} {json.dumps(data, ensure_ascii=False)}")
    snapshot_task_state(task, message, step_key)


def finalize_task_artifacts(task: dict[str, Any], metadata: dict[str, Any]) -> None:
    artifact_paths = metadata.get("artifacts", {})
    task["artifact_paths"] = {
        "ocr_markdown": artifact_paths.get("ocr_markdown"),
        "ocr_assets": artifact_paths.get("ocr_assets"),
        "slides_pdf": artifact_paths.get("slides_pdf"),
        "slides_tex": artifact_paths.get("slides_tex"),
        "script": artifact_paths.get("script"),
        "speech_manifest": artifact_paths.get("speech_manifest"),
        "audio_transcript": artifact_paths.get("audio_transcript"),
        "agentic_pacing": artifact_paths.get("agentic_pacing"),
        "cursor": artifact_paths.get("cursor"),
        "subtitles": artifact_paths.get("subtitles"),
        "video": artifact_paths.get("video"),
        "sat": str(Path(task["result_dir"]) / "sat.json"),
        "token": str(Path(task["result_dir"]) / "token.json"),
    }
    task["artifact_paths"] = {key: value for key, value in task["artifact_paths"].items() if value}
    task["sample_video_url"] = f"/api/tasks/{task['id']}/artifacts/video"
    task["artifacts"] = {
        key: f"/api/tasks/{task['id']}/artifacts/{key}"
        for key in task["artifact_paths"]
    }


def build_pipeline_command(task: dict[str, Any]) -> list[str]:
    settings = task["settings"]
    python_exe = str(resolve_pipeline_python())
    ref_audio = ROOT.parent / "assets" / "demo" / "reference.wav"
    ref_text = "Some call me nature, others call me mother nature."
    avatar_image = ROOT / "avatar" / "kafka.jpg"
    command = [
        python_exe,
        str(PIPELINE_SCRIPT),
        "--paper_pdf",
        task["upload_path"],
        "--result_dir",
        task["result_dir"],
        "--desired_minutes",
        str(task["desired_minutes"]),
        "--target_slides",
        str(task.get("target_slide_count") or 12),
        "--goal_prompt",
        (
            f"{settings['system_prompt']}\n\n"
            f"User goal: {task['goal_prompt']}\n"
            f"Style: {task['preferred_slide_style']}\n"
            f"Agentic framework: {task.get('agentic_framework', 'langgraph')}"
        ),
        "--model",
        settings["text_model"],
        "--ollama_url",
        settings["ollama_url"],
        "--temperature",
        str(settings["temperature"]),
        "--top_p",
        str(settings["top_p"]),
        "--mineru_method",
        "ocr",
        "--ref_audio",
        str(ref_audio),
        "--ref_text",
        ref_text,
    ]
    avatar_mode = task.get("avatar_mode", "none")
    if avatar_mode != "none":
        command.extend(["--avatar_mode", avatar_mode, "--avatar_position", task.get("avatar_position", "bottom_right")])
        if avatar_image.exists():
            command.extend(["--avatar_image", str(avatar_image)])
    return command


def apply_pipeline_event(task: dict[str, Any], payload: dict[str, Any]) -> None:
    pipeline_step = payload.get("step")
    ui_step = PIPELINE_STEP_MAP.get(pipeline_step)
    message = str(payload.get("message", "Pipeline event."))
    data = payload.get("data", {})
    if payload.get("kind") == "start" and ui_step:
        mark_step_started(task, ui_step, message)
    elif payload.get("kind") == "done" and ui_step:
        mark_step_completed(task, ui_step, message, data if isinstance(data, dict) else {})
    elif pipeline_step == "pipeline":
        append_log(task, message)
        snapshot_task_state(task, message)


def process_task(task_id: str) -> None:
    task = read_task(task_id)
    task["status"] = "running"
    task["started_at"] = now_ts()
    task["job"]["state"] = "running"
    task["job"]["worker_id"] = WORKER_ID
    task["job"]["attempt"] += 1
    task["job"]["started_at"] = task["started_at"]
    task["job"]["last_heartbeat"] = task["started_at"]
    append_log(task, f"Worker {WORKER_ID} accepted job.")
    snapshot_task_state(task, "Worker accepted job.")
    write_task(task)
    refresh_queue_positions()

    command = build_pipeline_command(task)
    stdout_path = Path(task["result_dir"]) / "pipeline_stdout.log"
    stderr_tail: list[str] = []
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    append_log(task, "Starting real pipeline subprocess.")
    snapshot_task_state(task, "Starting real pipeline subprocess.")
    write_task(task)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=str(ROOT.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    with stdout_path.open("w", encoding="utf-8") as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            stripped = line.strip()
            if stripped.startswith("PIPELINE_EVENT "):
                payload = json.loads(stripped.removeprefix("PIPELINE_EVENT "))
                task = read_task(task_id)
                apply_pipeline_event(task, payload)
                task["job"]["last_heartbeat"] = now_ts()
                write_task(task)
            elif stripped:
                stderr_tail.append(stripped)
                stderr_tail = stderr_tail[-20:]

    return_code = process.wait()
    task = read_task(task_id)
    if return_code != 0:
        raise RuntimeError("Real pipeline failed with exit code " + str(return_code) + ":\n" + "\n".join(stderr_tail[-8:]))

    sat_path = Path(task["result_dir"]) / "sat.json"
    if not sat_path.exists():
        raise RuntimeError(f"Pipeline completed without sat.json at {sat_path}")
    metadata = json.loads(sat_path.read_text(encoding="utf-8"))
    total_task_ticks = sum(int(step.get("tick_total", 0)) for step in task["steps"])

    task["status"] = "completed"
    task["finished_at"] = now_ts()
    task["progress"] = 1.0
    task["progress_counts"] = {"completed": total_task_ticks, "total": total_task_ticks}
    task["job"]["state"] = "completed"
    task["job"]["finished_at"] = task["finished_at"]
    task["job"]["last_heartbeat"] = task["finished_at"]
    finalize_task_artifacts(task, metadata)
    append_log(task, "Real pipeline completed. Output artifacts attached.")
    snapshot_task_state(task, "Pipeline completed.")
    write_task(task)
    refresh_queue_positions()


def worker_loop() -> None:
    while True:
        next_task_id = None
        queued = queued_tasks()
        if queued:
            next_task_id = queued[0]["id"]
        else:
            QUEUE_EVENT.clear()
            QUEUE_EVENT.wait(timeout=1)
            continue

        try:
            process_task(next_task_id)
        except Exception as exc:
            task = read_task(next_task_id)
            task["status"] = "failed"
            task["finished_at"] = now_ts()
            task["job"]["state"] = "failed"
            task["job"]["finished_at"] = task["finished_at"]
            task["job"]["last_heartbeat"] = task["finished_at"]
            append_log(task, f"Worker failed: {exc}")
            snapshot_task_state(task, f"Worker failed: {exc}")
            write_task(task)
            refresh_queue_positions()


def ensure_worker() -> None:
    if os.environ.get("WEB_DISABLE_WORKER_THREAD") == "1":
        return
    global WORKER_THREAD
    with WORKER_LOCK:
        if WORKER_THREAD is not None and WORKER_THREAD.is_alive():
            return
        WORKER_THREAD = threading.Thread(target=worker_loop, name="control-room-worker", daemon=True)
        WORKER_THREAD.start()


def list_uploads() -> list[dict[str, Any]]:
    uploads: list[dict[str, Any]] = []
    for meta_path in sorted(UPLOAD_DIR.glob("*.json"), reverse=True):
        uploads.append(json.loads(meta_path.read_text(encoding="utf-8")))
    return uploads[:20]


app = FastAPI(title="Control Room", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/aimooc/assets", StaticFiles(directory=AIMOOC_STATIC_DIR / "assets"), name="aimooc-assets")


@app.on_event("startup")
def startup_worker() -> None:
    db_connect().close()
    if os.environ.get("WEB_DISABLE_WORKER_THREAD") == "1":
        return
    ensure_worker()
    refresh_queue_positions()
    if queued_tasks():
        QUEUE_EVENT.set()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/history", response_class=HTMLResponse)
def history_page() -> str:
    return (STATIC_DIR / "history.html").read_text(encoding="utf-8")


@app.get("/replay/{task_id}", response_class=HTMLResponse)
def replay_page(task_id: str) -> str:
    path = STATIC_DIR / "replay.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Replay page missing")
    return path.read_text(encoding="utf-8")


@app.get("/aimooc", response_class=HTMLResponse)
def aimooc_page() -> str:
    path = AIMOOC_STATIC_DIR / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="AIMOOC UI has not been built")
    return path.read_text(encoding="utf-8")


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return load_settings()


@app.post("/api/settings")
def post_settings(payload: SettingsPayload) -> dict[str, Any]:
    return save_settings(payload.model_dump())


@app.post("/api/settings/test-ollama")
def test_ollama(payload: OllamaTestPayload) -> dict[str, Any]:
    return probe_ollama(payload.ollama_url)

@app.get("/api/ollama/models")
def get_ollama_models() -> dict[str, Any]:
    settings = load_settings()
    return probe_ollama(settings["ollama_url"])


@app.get("/api/slide-styles")
def get_slide_styles() -> list[dict[str, Any]]:
    return SLIDE_STYLE_TEMPLATES

@app.get("/api/tool-catalog")
def get_tool_catalog() -> list[dict[str, Any]]:
    return tool_catalog()


@app.get("/api/tools/manifest")
def get_tools_manifest() -> dict[str, Any]:
    return {
        "manifest_path": str(TOOLS_MANIFEST),
        "tools": tool_catalog(),
    }


@app.get("/api/agent-catalog")
def get_agent_catalog() -> list[dict[str, Any]]:
    return agent_catalog()


@app.get("/api/agent-graph")
def get_agent_graph() -> dict[str, Any]:
    return agentic_graph_status(AGENTS_MANIFEST, TOOLS_MANIFEST)


@app.get("/api/agent-frameworks")
def get_agent_frameworks() -> list[dict[str, Any]]:
    return available_frameworks()


@app.get("/api/agents/{agent_key}/skills.md")
def get_agent_skills(agent_key: str) -> FileResponse:
    path = resolve_agent_skills_md(agent_key)
    return FileResponse(str(path), media_type="text/markdown", filename=path.name)

@app.put("/api/agents/{agent_key}/skills.md")
def update_agent_skills(agent_key: str, payload: SkillsUpdatePayload) -> dict[str, Any]:
    path = resolve_agent_skills_md(agent_key)
    content = payload.content.rstrip() + "\n"
    with SKILL_EDIT_LOCK:
        path.write_text(content, encoding="utf-8")
    return {
        "ok": True,
        "agent_key": agent_key,
        "path": str(path),
        "bytes": len(content.encode("utf-8")),
    }

@app.get("/api/queue")
def get_queue() -> dict[str, Any]:
    refresh_queue_positions()
    return queue_state()


@app.post("/api/upload")
async def upload_paper(file: UploadFile = File(...)) -> dict[str, Any]:
    upload_id = str(uuid.uuid4())
    safe_name = file.filename or "paper.bin"
    ext = Path(safe_name).suffix
    file_path = UPLOAD_DIR / f"{upload_id}{ext}"
    meta_path = UPLOAD_DIR / f"{upload_id}.json"
    content = await file.read()
    file_path.write_bytes(content)
    payload = {
        "id": upload_id,
        "file_name": safe_name,
        "saved_path": str(file_path),
        "size_bytes": len(content),
        "uploaded_at": now_ts(),
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


@app.post("/api/uploads/batch")
async def upload_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    uploaded = []
    for file in files:
        uploaded.append(await upload_paper(file))
    return {"count": len(uploaded), "uploads": uploaded}


@app.get("/api/uploads")
def get_uploads() -> list[dict[str, Any]]:
    return list_uploads()


def source_selection_payload(payload: CreateAIMOOCProjectPayload) -> list[AIMOOCSourceSelection]:
    if payload.sources:
        return payload.sources
    selections: list[AIMOOCSourceSelection] = []
    for index, source_id in enumerate(payload.source_ids):
        selections.append(
            AIMOOCSourceSelection(
                source_id=source_id,
                role="primary" if index == 0 else "reference",
                priority=1 if index == 0 else 3,
            )
        )
    return selections


def resolve_task_artifact_path(task_id: str, artifact_key: str = "video") -> Path:
    task = read_task(task_id)
    artifact_path = task.get("artifact_paths", {}).get(artifact_key)
    if not artifact_path:
        raise HTTPException(status_code=404, detail=f"Task artifact not found: {artifact_key}")
    path = Path(artifact_path)
    if not path.exists() or path.is_dir():
        raise HTTPException(status_code=404, detail=f"Task artifact file missing: {artifact_key}")
    return path


def load_upload_meta(upload_id: str) -> dict[str, Any]:
    upload_meta = UPLOAD_DIR / f"{upload_id}.json"
    if not upload_meta.exists():
        raise HTTPException(status_code=404, detail=f"Upload not found: {upload_id}")
    return json.loads(upload_meta.read_text(encoding="utf-8"))


@app.post("/api/aimooc/projects")
def create_aimooc_project(payload: CreateAIMOOCProjectPayload) -> dict[str, Any]:
    selections = source_selection_payload(payload)
    if not selections:
        raise HTTPException(status_code=422, detail="At least one source is required")
    project_id = str(uuid.uuid4())
    result_dir = AIMOOC_RESULT_DIR / project_id
    source_items: list[SourceItem] = []
    for selection in selections:
        item = build_source_item(
            load_upload_meta(selection.source_id),
            role=selection.role,
            priority=selection.priority,
            notes=selection.notes,
        )
        if selection.title:
            item.title = selection.title
        source_items.append(item)
    manifest = build_source_manifest(project_id, source_items)
    framework = normalize_framework(payload.agentic_framework)
    spec = AIMOOCSpec(
        course_title=payload.course_title,
        audience=payload.audience,
        learning_objectives=payload.learning_objectives,
        requirements=payload.requirements,
        total_minutes=payload.total_minutes,
        module_count=payload.module_count,
        lessons_per_module=payload.lessons_per_module,
        preferred_style=payload.preferred_style,
        language=payload.language,
        difficulty=payload.difficulty,
        include_quizzes=payload.include_quizzes,
        include_assignments=payload.include_assignments,
        include_avatar=payload.include_avatar,
        avatar_mode=payload.avatar_mode,
        feedback_mode=payload.feedback_mode,
        agentic_framework=framework,
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_path = result_dir / "source_manifest.json"
    course_spec_path = result_dir / "course_spec.json"
    write_model(source_manifest_path, manifest)
    write_model(course_spec_path, spec)
    lesson_video_source = None
    if payload.render_videos and payload.lesson_video_task_id:
        lesson_video_source = resolve_task_artifact_path(payload.lesson_video_task_id, "video")
    avatar_image = ROOT / "avatar" / "kafka.jpg"
    package = run_aimooc_pipeline(
        source_manifest_path,
        course_spec_path,
        result_dir,
        render_media=payload.render_videos,
        lesson_video_source=lesson_video_source,
        avatar_image=avatar_image if avatar_image.exists() else None,
    )
    stored_payload = {
        "source_manifest": manifest.model_dump(),
        "course_spec": spec.model_dump(),
        "package": package,
        "render_videos": payload.render_videos,
        "lesson_video_task_id": payload.lesson_video_task_id,
    }
    upsert_project(project_id, spec.course_title, "completed", framework, result_dir, stored_payload)
    add_project_version(project_id, "v001_initial", result_dir / "versions" / "v001_initial")
    return get_aimooc_project(project_id)


@app.get("/api/aimooc/projects")
def list_aimooc_projects() -> list[dict[str, Any]]:
    with db_connect() as conn:
        rows = conn.execute("SELECT * FROM aimooc_projects ORDER BY updated_at DESC LIMIT 50").fetchall()
    return [project_summary(row) for row in rows]


@app.get("/api/aimooc/projects/{project_id}")
def get_aimooc_project(project_id: str) -> dict[str, Any]:
    row = project_row(project_id)
    payload = json.loads(row["payload_json"])
    result_dir = Path(row["result_dir"])
    package_path = result_dir / "course_package_manifest.json"
    course_plan_path = result_dir / "course_plan.json"
    return {
        **project_summary(row),
        "payload": payload,
        "package_manifest": json.loads(package_path.read_text(encoding="utf-8")) if package_path.exists() else None,
        "course_plan": json.loads(course_plan_path.read_text(encoding="utf-8")) if course_plan_path.exists() else None,
    }


@app.post("/api/aimooc/projects/{project_id}/feedback")
def create_aimooc_feedback(project_id: str, payload: AIMOOCFeedbackPayload) -> dict[str, Any]:
    row = project_row(project_id)
    project_dir = Path(row["result_dir"])
    feedback_round = write_feedback_round(project_dir, payload.base_version, payload.feedback)
    with db_connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO aimooc_feedback(project_id, round_id, base_version, payload_json, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (
                project_id,
                feedback_round.round_id,
                feedback_round.base_version,
                feedback_round.model_dump_json(),
                feedback_round.created_at,
            ),
        )
        conn.commit()
    return feedback_round.model_dump()


@app.post("/api/aimooc/projects/{project_id}/revise")
def revise_aimooc_project(project_id: str, payload: AIMOOCFeedbackPayload) -> dict[str, Any]:
    row = project_row(project_id)
    project_dir = Path(row["result_dir"])
    feedback_round = write_feedback_round(project_dir, payload.base_version, payload.feedback)
    revision = revise_project(project_dir, payload.base_version, feedback_round)
    add_project_version(project_id, revision.version_id, project_dir / "versions" / revision.version_id)
    with db_connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO aimooc_feedback(project_id, round_id, base_version, payload_json, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (project_id, feedback_round.round_id, feedback_round.base_version, feedback_round.model_dump_json(), feedback_round.created_at),
        )
        conn.execute(
            "UPDATE aimooc_projects SET status=?, updated_at=? WHERE project_id=?",
            ("revised", now_ts(), project_id),
        )
        conn.commit()
    return {"feedback_round": feedback_round.model_dump(), "revision": revision.model_dump()}


@app.get("/api/aimooc/projects/{project_id}/versions")
def list_aimooc_versions(project_id: str) -> list[dict[str, Any]]:
    project_row(project_id)
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT version_id, path, created_at FROM aimooc_versions WHERE project_id=? ORDER BY created_at ASC",
            (project_id,),
        ).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/aimooc/projects/{project_id}/artifacts/{artifact_path:path}")
def get_aimooc_artifact(project_id: str, artifact_path: str) -> FileResponse:
    row = project_row(project_id)
    root = Path(row["result_dir"]).resolve()
    artifact = (root / artifact_path).resolve()
    if not str(artifact).startswith(str(root)) or not artifact.exists() or artifact.is_dir():
        raise HTTPException(status_code=404, detail="AIMOOC artifact not found")
    media_type = "application/octet-stream"
    if artifact.suffix == ".json":
        media_type = "application/json"
    elif artifact.suffix == ".mp4":
        media_type = "video/mp4"
    elif artifact.suffix in {".md", ".txt", ".srt"}:
        media_type = "text/plain"
    return FileResponse(str(artifact), media_type=media_type, filename=artifact.name)


@app.post("/api/tasks")
def create_task(payload: CreateTaskPayload) -> dict[str, Any]:
    ensure_worker()
    upload_meta = UPLOAD_DIR / f"{payload.upload_id}.json"
    if not upload_meta.exists():
        raise HTTPException(status_code=404, detail="Upload not found")

    upload_info = json.loads(upload_meta.read_text(encoding="utf-8"))
    settings = load_settings()
    steps = build_steps(payload, settings)
    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "status": "queued",
        "created_at": now_ts(),
        "goal_prompt": payload.goal_prompt,
        "desired_minutes": payload.desired_minutes,
        "target_slide_count": payload.target_slide_count,
        "preferred_slide_style": payload.preferred_slide_style,
        "agentic_framework": normalize_framework(payload.agentic_framework),
        "avatar_mode": payload.avatar_mode,
        "avatar_position": payload.avatar_position,
        "upload_id": payload.upload_id,
        "upload_name": upload_info["file_name"],
        "upload_path": upload_info["saved_path"],
        "result_dir": str((WEB_RESULT_DIR / task_id).resolve()),
        "settings": settings,
        "steps": steps,
    }
    initialize_task_runtime(task)
    write_task(task)
    refresh_queue_positions()
    QUEUE_EVENT.set()
    return task


@app.get("/api/tasks")
def get_tasks() -> list[dict[str, Any]]:
    items = sorted(all_tasks(), key=lambda task: task.get("created_at", 0.0), reverse=True)
    return [task_summary(task) for task in items[:50]]


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    return read_task(task_id)


@app.get("/api/tasks/{task_id}/timeline")
def get_task_timeline(task_id: str) -> dict[str, Any]:
    task = read_task(task_id)
    return {
        "task_id": task["id"],
        "upload_name": task["upload_name"],
        "status": task["status"],
        "timeline": task.get("timeline", []),
    }


@app.get("/api/tasks/{task_id}/ocr-assets")
def get_task_ocr_assets(task_id: str) -> dict[str, Any]:
    task = read_task(task_id)
    manifest_path = task.get("artifact_paths", {}).get("ocr_assets")
    if not manifest_path:
        return {"assets": [], "counts": {}}
    path = Path(manifest_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="OCR assets missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for asset in manifest.get("assets", []):
        if asset.get("image"):
            asset["image_url"] = f"/api/tasks/{task_id}/ocr-assets/{asset['id']}/image"
    return manifest


@app.get("/api/tasks/{task_id}/ocr-assets/{asset_id}/image")
def get_task_ocr_asset_image(task_id: str, asset_id: str) -> FileResponse:
    manifest = get_task_ocr_assets(task_id)
    for asset in manifest.get("assets", []):
        if asset.get("id") == asset_id and asset.get("image"):
            path = Path(asset["image"]).resolve()
            if not path.exists():
                raise HTTPException(status_code=404, detail="OCR image missing")
            return FileResponse(str(path), media_type="image/jpeg", filename=path.name)
    raise HTTPException(status_code=404, detail="OCR asset not found")


@app.get("/api/tasks/{task_id}/artifacts/{artifact_name}")
def get_task_artifact(task_id: str, artifact_name: str) -> FileResponse:
    artifact = resolve_task_artifact(task_id, artifact_name)
    media_type = "application/octet-stream"
    if artifact.suffix == ".pdf":
        media_type = "application/pdf"
    elif artifact.suffix in {".txt", ".json", ".tex", ".srt"}:
        media_type = "text/plain"
    elif artifact.suffix == ".mp4":
        media_type = "video/mp4"
    return FileResponse(str(artifact), media_type=media_type, filename=artifact.name)


@app.get("/api/artifacts/{artifact_name}")
def get_artifact(artifact_name: str) -> FileResponse:
    artifact = resolve_artifact(artifact_name)
    media_type = "application/octet-stream"
    if artifact.suffix == ".pdf":
        media_type = "application/pdf"
    elif artifact.suffix in {".txt", ".json"}:
        media_type = "text/plain"
    elif artifact.suffix == ".mp4":
        media_type = "video/mp4"
    return FileResponse(str(artifact), media_type=media_type, filename=artifact.name)


@app.get("/api/sample-video")
def sample_video() -> FileResponse:
    sample = resolve_sample_video()
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample video missing")
    return FileResponse(str(sample), media_type="video/mp4", filename=sample.name)


@app.get("/api/health")
def health() -> JSONResponse:
    settings = load_settings()
    refresh_queue_positions()
    return JSONResponse(
        {
            "ok": True,
            "sample_video_exists": resolve_sample_video() is not None,
            "task_count": len(list(TASK_DIR.glob("*.json"))),
            "ollama_url": settings["ollama_url"],
            "gpu": gpu_status(),
            "pipeline_python": str(resolve_pipeline_python()),
            "agentic_graph": agentic_graph_status(AGENTS_MANIFEST, TOOLS_MANIFEST),
            "agentic_frameworks": available_frameworks(),
            "queue": queue_state(),
        }
    )




