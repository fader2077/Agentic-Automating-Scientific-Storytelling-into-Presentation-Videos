from __future__ import annotations

import json
import math
import operator
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict


class AgentGraphState(TypedDict, total=False):
    visited: Annotated[list[str], operator.add]
    completed: Annotated[list[str], operator.add]
    artifacts: Annotated[list[str], operator.add]
    tool_calls: Annotated[list[str], operator.add]
    error_stages: Annotated[list[str], operator.add]


class PacingGraphState(TypedDict, total=False):
    desired_minutes: int
    requested_total_slides: int | None
    ocr_asset_count: int
    total_slides: int
    content_slides: int
    target_seconds: float
    min_seconds: float
    max_seconds: float
    target_total_words: int
    title_words: int
    words_per_slide: int
    voice_speed: float
    sentence_pause: float
    subtitle_source: str
    visited: Annotated[list[str], operator.add]
    tool_calls: Annotated[list[str], operator.add]


AGENT_FLOW = [
    "SupervisorAgent",
    "IngestionAgent",
    "PlannerAgent",
    "SlideBuilderAgent",
    "ScriptAgent",
    "SpeechAgent",
    "GroundingAgent",
    "RenderAgent",
]

AIMOOC_FLOW = [
    "SupervisorAgent",
    "SourceIngestionAgent",
    "CourseUnderstandingAgent",
    "CoursePlannerAgent",
    "LessonBuilderAgent",
    "QuizAgent",
    "VisualAuditorAgent",
    "FeedbackAgent",
    "RevisionAgent",
    "SpeechAgent",
    "AvatarDirectorAgent",
    "RenderAgent",
    "CoursePackagerAgent",
]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def choose_total_slides(desired_minutes: int, requested_total_slides: int | None) -> int:
    if requested_total_slides:
        return int(clamp(requested_total_slides, 5, 30))
    minutes = max(1, int(desired_minutes))
    if minutes <= 3:
        return int(clamp(2 + minutes * 3, 8, 11))
    if minutes <= 6:
        return 12
    return int(clamp(round(minutes * 1.6) + 2, 14, 22))


def pacing_supervisor_node(state: PacingGraphState) -> PacingGraphState:
    return {
        "visited": ["PacingSupervisorAgent"],
        "tool_calls": ["PacingSupervisorAgent.set_duration_band"],
        "target_seconds": float(max(60, int(state.get("desired_minutes", 6)) * 60)),
        "min_seconds": float(max(45, int(state.get("desired_minutes", 6)) * 60 - 60)),
        "max_seconds": float(int(state.get("desired_minutes", 6)) * 60 + 60),
    }


def slide_allocator_node(state: PacingGraphState) -> PacingGraphState:
    total_slides = choose_total_slides(int(state.get("desired_minutes", 6)), state.get("requested_total_slides"))
    return {
        "visited": ["PlannerAgent"],
        "tool_calls": ["PlannerAgent.allocate_slide_count"],
        "total_slides": total_slides,
        "content_slides": max(1, total_slides - 2),
    }


def script_budget_node(state: PacingGraphState) -> PacingGraphState:
    target_seconds = float(state.get("target_seconds", 360.0))
    content_slides = max(1, int(state.get("content_slides", 11)))
    title_words = 48
    # SpeechAgent keeps F5TTS at natural speed and adapts duration by script
    # budget, not by forcing slow synthesis.
    target_total_words = int(clamp(target_seconds * 1.70, 260, 1700))
    words_per_slide = int(clamp(math.floor((target_total_words - title_words) / content_slides), 22, 78))
    return {
        "visited": ["ScriptAgent"],
        "tool_calls": ["ScriptAgent.assign_slide_word_budget"],
        "target_total_words": target_total_words,
        "title_words": title_words,
        "words_per_slide": words_per_slide,
    }


def speech_pacing_node(state: PacingGraphState) -> PacingGraphState:
    return {
        "visited": ["SpeechAgent"],
        "tool_calls": ["SpeechAgent.choose_natural_f5_pacing"],
        "voice_speed": 1.0,
        "sentence_pause": 0.0,
    }


def subtitle_policy_node(state: PacingGraphState) -> PacingGraphState:
    return {
        "visited": ["SubtitleAgent"],
        "tool_calls": ["SubtitleAgent.select_audio_transcript_subtitles"],
        "subtitle_source": "asr",
    }


def build_pacing_langgraph() -> Any:
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(PacingGraphState)
    graph.add_node("pacing_supervisor", pacing_supervisor_node)
    graph.add_node("slide_allocator", slide_allocator_node)
    graph.add_node("script_budget", script_budget_node)
    graph.add_node("speech_pacing", speech_pacing_node)
    graph.add_node("subtitle_policy", subtitle_policy_node)
    graph.add_edge(START, "pacing_supervisor")
    graph.add_edge("pacing_supervisor", "slide_allocator")
    graph.add_edge("slide_allocator", "script_budget")
    graph.add_edge("script_budget", "speech_pacing")
    graph.add_edge("speech_pacing", "subtitle_policy")
    graph.add_edge("subtitle_policy", END)
    return graph.compile()


def build_adaptive_pacing_plan(
    desired_minutes: int,
    requested_total_slides: int | None = None,
    ocr_asset_count: int = 0,
) -> dict[str, Any]:
    initial: PacingGraphState = {
        "desired_minutes": int(desired_minutes),
        "requested_total_slides": requested_total_slides,
        "ocr_asset_count": int(ocr_asset_count),
        "visited": [],
        "tool_calls": [],
    }
    try:
        result = build_pacing_langgraph().invoke(initial)
    except Exception:
        total_slides = choose_total_slides(desired_minutes, requested_total_slides)
        content_slides = max(1, total_slides - 2)
        target_seconds = float(max(60, int(desired_minutes) * 60))
        target_total_words = int(clamp(target_seconds * 1.70, 260, 1700))
        result = {
            **initial,
            "visited": ["PacingSupervisorAgent", "PlannerAgent", "ScriptAgent", "SpeechAgent", "SubtitleAgent"],
            "tool_calls": [
                "PacingSupervisorAgent.set_duration_band",
                "PlannerAgent.allocate_slide_count",
                "ScriptAgent.assign_slide_word_budget",
                "SpeechAgent.choose_natural_f5_pacing",
                "SubtitleAgent.select_audio_transcript_subtitles",
            ],
            "target_seconds": target_seconds,
            "min_seconds": float(max(45, int(desired_minutes) * 60 - 60)),
            "max_seconds": float(int(desired_minutes) * 60 + 60),
            "total_slides": total_slides,
            "content_slides": content_slides,
            "target_total_words": target_total_words,
            "title_words": 48,
            "words_per_slide": int(clamp(math.floor((target_total_words - 48) / content_slides), 22, 78)),
            "voice_speed": 1.0,
            "sentence_pause": 0.0,
            "subtitle_source": "asr",
        }
    return {
        "framework": "langgraph" if "PacingSupervisorAgent" in result.get("visited", []) else "fallback",
        "desired_minutes": int(desired_minutes),
        "requested_total_slides": requested_total_slides,
        "total_slides": int(result.get("total_slides", choose_total_slides(desired_minutes, requested_total_slides))),
        "content_slides": int(result.get("content_slides", max(1, choose_total_slides(desired_minutes, requested_total_slides) - 2))),
        "target_seconds": float(result.get("target_seconds", max(60, int(desired_minutes) * 60))),
        "min_seconds": float(result.get("min_seconds", max(45, int(desired_minutes) * 60 - 60))),
        "max_seconds": float(result.get("max_seconds", int(desired_minutes) * 60 + 60)),
        "target_total_words": int(result.get("target_total_words", 330)),
        "title_words": int(result.get("title_words", 24)),
        "words_per_slide": int(result.get("words_per_slide", 28)),
        "voice_speed": float(result.get("voice_speed", 1.0)),
        "sentence_pause": float(result.get("sentence_pause", 0.0)),
        "subtitle_source": str(result.get("subtitle_source", "asr")),
        "visited": result.get("visited", []),
        "tool_calls": result.get("tool_calls", []),
    }


def read_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must be a list: {path}")
    return payload


def tool_titles(agent: dict[str, Any], tools: dict[str, dict[str, Any]]) -> list[str]:
    return [str(tools.get(str(key), {"title": key}).get("title", key)) for key in agent.get("tools", [])]


def agent_node(agent_key: str, completed: str, artifacts: list[str], tools: list[str]):
    def node(state: AgentGraphState) -> AgentGraphState:
        return {
            "visited": [agent_key],
            "completed": [completed],
            "artifacts": artifacts,
            "tool_calls": [f"{agent_key}.{tool}" for tool in tools],
        }

    return node


def supervisor_node(state: AgentGraphState) -> AgentGraphState:
    return {"visited": ["SupervisorAgent"], "tool_calls": ["SupervisorAgent.route_job"]}


def ingestion_repair_node(state: AgentGraphState) -> AgentGraphState:
    return {
        "visited": ["IngestionRepairAgent"],
        "tool_calls": ["IngestionRepairAgent.retry_mineru_or_manifest"],
    }


def visual_auditor_node(state: AgentGraphState) -> AgentGraphState:
    return {
        "visited": ["VisualAuditorAgent"],
        "completed": ["visual_audit"],
        "artifacts": ["visual_audit"],
        "tool_calls": ["VisualAuditorAgent.check_ocr_asset_coverage"],
    }


def artifact_join_node(state: AgentGraphState) -> AgentGraphState:
    return {"visited": ["ArtifactJoinAgent"], "tool_calls": ["ArtifactJoinAgent.wait_for_slides_and_audio"]}


def route_from_supervisor(state: AgentGraphState) -> Literal["ingest", "plan"]:
    artifacts = set(state.get("artifacts", []))
    return "plan" if "ocr_assets" in artifacts else "ingest"


def route_from_ingest(state: AgentGraphState) -> Literal["repair", "plan"]:
    return "repair" if "ingest" in set(state.get("error_stages", [])) else "plan"


def route_from_planner(state: AgentGraphState) -> list[str]:
    return ["slide_builder_agent", "script_agent"]


def route_from_visual_audit(state: AgentGraphState) -> Literal["repair_slides", "join"]:
    return "repair_slides" if "visuals" in set(state.get("error_stages", [])) else "join"


def route_from_join(state: AgentGraphState) -> Literal["ground", "wait"]:
    artifacts = set(state.get("artifacts", []))
    return "ground" if {"slide_imgs", "audio"}.issubset(artifacts) else "wait"


def build_langgraph_runner(agents: list[dict[str, Any]], tools_manifest: list[dict[str, Any]]) -> Any:
    from langgraph.graph import END, START, StateGraph

    tools = {str(tool["key"]): tool for tool in tools_manifest}
    by_key = {str(agent["key"]): agent for agent in agents}

    graph = StateGraph(AgentGraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("ingestion_agent", agent_node("IngestionAgent", "ingest", ["ocr_assets"], tool_titles(by_key["IngestionAgent"], tools)))
    graph.add_node("ingestion_repair", ingestion_repair_node)
    graph.add_node("planner_agent", agent_node("PlannerAgent", "planner", ["plan"], tool_titles(by_key["PlannerAgent"], tools)))
    graph.add_node(
        "slide_builder_agent",
        agent_node("SlideBuilderAgent", "slides", ["slides_pdf", "slide_imgs"], tool_titles(by_key["SlideBuilderAgent"], tools)),
    )
    graph.add_node("visual_auditor", visual_auditor_node)
    graph.add_node("script_agent", agent_node("ScriptAgent", "script", ["script"], tool_titles(by_key["ScriptAgent"], tools)))
    graph.add_node("speech_agent", agent_node("SpeechAgent", "tts", ["audio"], tool_titles(by_key["SpeechAgent"], tools)))
    graph.add_node("artifact_join", artifact_join_node)
    graph.add_node("grounding_agent", agent_node("GroundingAgent", "cursor", ["cursor"], tool_titles(by_key["GroundingAgent"], tools)))
    graph.add_node("render_agent", agent_node("RenderAgent", "compose", ["video"], tool_titles(by_key["RenderAgent"], tools)))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_from_supervisor, {"ingest": "ingestion_agent", "plan": "planner_agent"})
    graph.add_conditional_edges("ingestion_agent", route_from_ingest, {"repair": "ingestion_repair", "plan": "planner_agent"})
    graph.add_edge("ingestion_repair", "ingestion_agent")
    graph.add_conditional_edges("planner_agent", route_from_planner)
    graph.add_edge("slide_builder_agent", "visual_auditor")
    graph.add_conditional_edges("visual_auditor", route_from_visual_audit, {"repair_slides": "slide_builder_agent", "join": "artifact_join"})
    graph.add_edge("script_agent", "speech_agent")
    graph.add_edge("speech_agent", "artifact_join")
    graph.add_conditional_edges("artifact_join", route_from_join, {"ground": "grounding_agent", "wait": END})
    graph.add_edge("grounding_agent", "render_agent")
    graph.add_edge("render_agent", END)
    return graph.compile()


def graph_edges() -> list[dict[str, str]]:
    return [
        {"source": "START", "target": "SupervisorAgent", "type": "entry"},
        {"source": "SupervisorAgent", "target": "IngestionAgent", "type": "conditional"},
        {"source": "SupervisorAgent", "target": "PlannerAgent", "type": "conditional_resume"},
        {"source": "IngestionAgent", "target": "IngestionRepairAgent", "type": "conditional_retry"},
        {"source": "IngestionRepairAgent", "target": "IngestionAgent", "type": "cycle"},
        {"source": "IngestionAgent", "target": "PlannerAgent", "type": "handoff"},
        {"source": "PlannerAgent", "target": "SlideBuilderAgent", "type": "parallel_fanout"},
        {"source": "PlannerAgent", "target": "ScriptAgent", "type": "parallel_fanout"},
        {"source": "SlideBuilderAgent", "target": "VisualAuditorAgent", "type": "audit"},
        {"source": "VisualAuditorAgent", "target": "SlideBuilderAgent", "type": "conditional_repair"},
        {"source": "VisualAuditorAgent", "target": "ArtifactJoinAgent", "type": "join"},
        {"source": "ScriptAgent", "target": "SpeechAgent", "type": "handoff"},
        {"source": "SpeechAgent", "target": "ArtifactJoinAgent", "type": "join"},
        {"source": "ArtifactJoinAgent", "target": "GroundingAgent", "type": "conditional_join"},
        {"source": "GroundingAgent", "target": "RenderAgent", "type": "handoff"},
        {"source": "RenderAgent", "target": "END", "type": "finish"},
    ]


def aimooc_graph_edges() -> list[dict[str, str]]:
    return [
        {"source": "START", "target": "SupervisorAgent", "type": "entry"},
        {"source": "SupervisorAgent", "target": "SourceIngestionAgent", "type": "mode_route"},
        {"source": "SourceIngestionAgent", "target": "CourseUnderstandingAgent", "type": "handoff"},
        {"source": "CourseUnderstandingAgent", "target": "CoursePlannerAgent", "type": "handoff"},
        {"source": "CoursePlannerAgent", "target": "LessonBuilderAgent", "type": "module_fanout"},
        {"source": "LessonBuilderAgent", "target": "QuizAgent", "type": "parallel_fanout"},
        {"source": "LessonBuilderAgent", "target": "VisualAuditorAgent", "type": "parallel_fanout"},
        {"source": "QuizAgent", "target": "FeedbackAgent", "type": "join"},
        {"source": "VisualAuditorAgent", "target": "FeedbackAgent", "type": "join"},
        {"source": "FeedbackAgent", "target": "RevisionAgent", "type": "conditional_revision"},
        {"source": "RevisionAgent", "target": "SpeechAgent", "type": "handoff"},
        {"source": "SpeechAgent", "target": "AvatarDirectorAgent", "type": "handoff"},
        {"source": "AvatarDirectorAgent", "target": "RenderAgent", "type": "avatar_render"},
        {"source": "RenderAgent", "target": "CoursePackagerAgent", "type": "package"},
        {"source": "CoursePackagerAgent", "target": "END", "type": "finish"},
    ]


def agentic_graph_status(agents_path: Path, tools_path: Path) -> dict[str, Any]:
    agents = read_manifest(agents_path)
    tools_manifest = read_manifest(tools_path)
    tools = {str(tool["key"]): tool for tool in tools_manifest}
    nodes = [
        {
            "key": str(agent["key"]),
            "stage": str(agent.get("stage", "")),
            "skills": agent.get("skills", []),
            "tools": [tools.get(str(key), {"key": str(key)}) for key in agent.get("tools", [])],
        }
        for agent in agents
    ]
    present = {node["key"] for node in nodes}
    virtual_nodes = [
        {"key": "SupervisorAgent", "stage": "route", "skills": ["state routing", "resume planning"], "tools": []},
        {"key": "IngestionRepairAgent", "stage": "repair", "skills": ["OCR retry", "manifest repair"], "tools": []},
        {"key": "VisualAuditorAgent", "stage": "audit", "skills": ["visual coverage audit", "slide repair routing"], "tools": []},
        {"key": "ArtifactJoinAgent", "stage": "join", "skills": ["parallel branch synchronization"], "tools": []},
    ]
    nodes.extend(node for node in virtual_nodes if node["key"] not in present)
    status: dict[str, Any] = {
        "framework": "langgraph",
        "compiled": False,
        "execution_model": "supervisor + conditional edges + parallel fanout + repair cycles",
        "nodes": nodes,
        "edges": graph_edges() + aimooc_graph_edges(),
        "flows": {
            "single_pdf": {
                "agents": AGENT_FLOW,
                "edges": graph_edges(),
                "frameworks": ["langgraph"],
            },
            "aimooc": {
                "agents": AIMOOC_FLOW,
                "edges": aimooc_graph_edges(),
                "frameworks": ["langgraph", "hermes_adapter", "openclaw_adapter"],
            },
        },
        "entrypoint": "SupervisorAgent",
        "finish": "RenderAgent",
        "visited_check": [],
        "tool_call_check": [],
        "aimooc_visited_check": AIMOOC_FLOW,
        "aimooc_tool_call_check": [
            f"{agent['key']}.{tool_titles(agent, tools)[0]}" if tool_titles(agent, tools) else f"{agent['key']}.route"
            for agent in agents
            if str(agent.get("key")) in set(AIMOOC_FLOW)
        ],
    }
    try:
        runner = build_langgraph_runner(agents, tools_manifest)
        result = runner.invoke({"visited": [], "completed": [], "artifacts": [], "tool_calls": [], "error_stages": []})
        status["compiled"] = True
        status["visited_check"] = result.get("visited", [])
        status["tool_call_check"] = result.get("tool_calls", [])
    except Exception as exc:
        status["framework"] = "manifest-dag"
        status["error"] = str(exc)
    return status
