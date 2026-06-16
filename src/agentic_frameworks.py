from __future__ import annotations

import importlib.util
import time
from typing import Any


FRAMEWORKS = [
    {
        "key": "langgraph",
        "title": "LangGraph",
        "description": "Existing compiled supervisor graph with conditional routing, fanout, join, and repair edges.",
        "installed": importlib.util.find_spec("langgraph") is not None,
        "mode": "native",
    },
    {
        "key": "hermes_adapter",
        "title": "Hermes Adapter",
        "description": "Hermes-compatible planner trace. Keeps the local pipeline deterministic while exposing Hermes-style delegated agent runs.",
        "installed": importlib.util.find_spec("hermes") is not None or importlib.util.find_spec("hermes_agent") is not None,
        "mode": "adapter",
    },
]


def available_frameworks() -> list[dict[str, Any]]:
    return [dict(item) for item in FRAMEWORKS]


def normalize_framework(raw: str | None) -> str:
    keys = {item["key"] for item in FRAMEWORKS}
    value = (raw or "langgraph").strip()
    return value if value in keys else "langgraph"


def run_agentic_trace(framework: str, project_id: str, stages: list[str]) -> dict[str, Any]:
    selected = normalize_framework(framework)
    if selected == "hermes_adapter":
        calls = [f"HermesDelegate.spawn({stage})" for stage in stages]
        execution_model = "hermes-compatible delegated agents"
    else:
        calls = [f"LangGraphNode.invoke({stage})" for stage in stages]
        execution_model = "langgraph supervisor graph"
    return {
        "project_id": project_id,
        "framework": selected,
        "execution_model": execution_model,
        "stages": stages,
        "tool_calls": calls,
        "created_at": time.time(),
    }
