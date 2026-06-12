from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict


class AgentGraphState(TypedDict, total=False):
    visited: Annotated[list[str], operator.add]
    completed: Annotated[list[str], operator.add]
    artifacts: Annotated[list[str], operator.add]
    tool_calls: Annotated[list[str], operator.add]
    error_stages: Annotated[list[str], operator.add]


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
        "edges": graph_edges(),
        "entrypoint": "SupervisorAgent",
        "finish": "RenderAgent",
        "visited_check": [],
        "tool_call_check": [],
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
