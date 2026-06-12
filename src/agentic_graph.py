from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict


class AgentGraphState(TypedDict, total=False):
    visited: list[str]
    current: str


def read_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must be a list: {path}")
    return payload


def ordered_edges(agents: list[dict[str, Any]]) -> list[dict[str, str]]:
    keys = [str(agent["key"]) for agent in agents]
    return [{"source": left, "target": right} for left, right in zip(keys, keys[1:])]


def build_langgraph_runner(agents: list[dict[str, Any]]) -> Any:
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentGraphState)

    for agent in agents:
        key = str(agent["key"])

        def node(state: AgentGraphState, agent_key: str = key) -> AgentGraphState:
            visited = list(state.get("visited", []))
            visited.append(agent_key)
            return {"visited": visited, "current": agent_key}

        graph.add_node(key, node)

    keys = [str(agent["key"]) for agent in agents]
    if not keys:
        raise ValueError("Agent graph needs at least one node.")
    graph.add_edge(START, keys[0])
    for left, right in zip(keys, keys[1:]):
        graph.add_edge(left, right)
    graph.add_edge(keys[-1], END)
    return graph.compile()


def agentic_graph_status(agents_path: Path, tools_path: Path) -> dict[str, Any]:
    agents = read_manifest(agents_path)
    tools = {str(tool["key"]): tool for tool in read_manifest(tools_path)}
    nodes = [
        {
            "key": str(agent["key"]),
            "stage": str(agent.get("stage", "")),
            "skills": agent.get("skills", []),
            "tools": [tools.get(str(key), {"key": str(key)}) for key in agent.get("tools", [])],
        }
        for agent in agents
    ]
    edges = ordered_edges(agents)
    status: dict[str, Any] = {
        "framework": "langgraph",
        "compiled": False,
        "nodes": nodes,
        "edges": edges,
        "entrypoint": str(agents[0]["key"]) if agents else "",
        "finish": str(agents[-1]["key"]) if agents else "",
        "visited_check": [],
    }
    try:
        runner = build_langgraph_runner(agents)
        result = runner.invoke({"visited": []})
        status["compiled"] = True
        status["visited_check"] = result.get("visited", [])
    except Exception as exc:
        status["framework"] = "manifest-dag"
        status["error"] = str(exc)
    return status
