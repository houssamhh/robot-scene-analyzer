from __future__ import annotations

import json
import os
from typing import Any, Dict

from mcp_tools import get_nearby_actors_from_yaml, get_robot_state_from_yaml
from perception import extract_scene_from_upload
from rag import build_simple_rag_context
from state import AppState

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False


PERCEPTION_TEXT_KEY = "scene_text"


def perception_node(state: AppState) -> AppState:
    return {
        **state,
        "perception_output": extract_scene_from_upload(
            state["point_cloud_bytes"], state["point_cloud_name"]
        ),
    }


def rag_node(state: AppState) -> AppState:
    question = state["question"]
    scene_text = state["perception_output"][PERCEPTION_TEXT_KEY]
    docs = build_simple_rag_context(question, scene_text)
    return {**state, "rag_output": docs}


def mcp_node(state: AppState) -> AppState:
    yaml_bytes = state.get("yaml_bytes", b"")
    if not yaml_bytes:
        return {
            **state,
            "mcp_output": {
                "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "roll_deg": 0.0, "yaw_deg": 0.0, "pitch_deg": 0.0},
                "speed_mps": 0.0,
                "robot_width_m": 0.55,
                "localization_confidence": 1.0,
                "planner_status": "unknown",
                "nearby_actors": [],
                "scenario_metadata": {"yaml_loaded": False, "num_actors": 0},
            },
        }

    robot_state = get_robot_state_from_yaml(yaml_bytes)
    nearby = get_nearby_actors_from_yaml(yaml_bytes)
    return {**state, "mcp_output": {**robot_state, **nearby}}


def rule_based_decision(state: AppState) -> Dict[str, Any]:
    scene = state["perception_output"]["scene"]
    free_space = scene.get("free_space", {})
    risk_flags = scene.get("risk_flags", [])
    right = free_space.get("right", 0)
    left = free_space.get("left", 0)
    front = free_space.get("front", 0)
    speed = float(state["mcp_output"].get("speed_mps", 0.0))
    dynamic_present = "dynamic_object_present" in risk_flags or any(
        o.get("motion") not in ("static", "none") for o in scene.get("objects", [])
    )

    if front < 0.25:
        risk = "High"
        recommendation = "Stop or perform conservative replanning before moving forward."
    elif front < 0.40 and max(left, right) > 0.70:
        open_side = "right" if right >= left else "left"
        risk = "Medium"
        recommendation = f"Slow down and replan through the more open {open_side} corridor."
    elif front < 0.40:
        risk = "Medium"
        recommendation = "Reduce speed and request a new local plan."
    else:
        risk = "Low"
        recommendation = "Proceed with caution."

    if dynamic_present and risk != "High":
        recommendation = "Slow down due to dynamic obstacle presence, then re-evaluate or replan."

    if speed > 2.0 and risk != "High":
        recommendation = "Reduce speed before attempting traversal, then re-evaluate or replan."

    evidence = [
        f"Front free-space ratio is {front}.",
        f"Left free-space ratio is {left}; right free-space ratio is {right}.",
    ]
    evidence.extend(
        [f"Retrieved rule: {doc['content']}" for doc in state.get("rag_output", [])[:2]]
    )
    evidence.append(
        f"Robot speed is {speed} m/s with localization confidence {state['mcp_output']['localization_confidence']}."
    )
    evidence.append(
        f"Scenario metadata reports {len(state['mcp_output'].get('nearby_actors', []))} nearby actors."
    )

    return {
        "risk_level": risk,
        "scene_assessment": state["perception_output"].get(
            "scene_summary", state["perception_output"][PERCEPTION_TEXT_KEY]
        ),
        "recommended_action": recommendation,
        "evidence": evidence,
    }


def llm_decision(state: AppState) -> Dict[str, Any]:
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a robotics decision-support coordinator. Return compact JSON with keys risk_level, scene_assessment, recommended_action, evidence. Base your answer only on the supplied perception summary, retrieved context, and robot state.",
            ),
            (
                "human",
                "Question: {question}\n\nPerception summary: {scene_text}\n\nRetrieved context: {retrieved}\n\nRobot state: {robot_state}",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke(
        {
            "question": state["question"],
            "scene_text": state["perception_output"][PERCEPTION_TEXT_KEY],
            "retrieved": json.dumps(state.get("rag_output", []), indent=2),
            "robot_state": json.dumps(state.get("mcp_output", {}), indent=2),
        }
    )
    return json.loads(raw)


def coordinator_node(state: AppState) -> AppState:
    if LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            final_output = llm_decision(state)
        except Exception:
            final_output = rule_based_decision(state)
    else:
        final_output = rule_based_decision(state)
    return {**state, "final_output": final_output}


def build_app_graph():
    if not LANGGRAPH_AVAILABLE:
        return None
    graph = StateGraph(AppState)
    graph.add_node("perception", perception_node)
    graph.add_node("rag", rag_node)
    graph.add_node("mcp", mcp_node)
    graph.add_node("coordinator", coordinator_node)
    graph.set_entry_point("perception")
    graph.add_edge("perception", "rag")
    graph.add_edge("rag", "mcp")
    graph.add_edge("mcp", "coordinator")
    graph.add_edge("coordinator", END)
    return graph.compile()


def run_pipeline(
    question: str,
    point_cloud_bytes: bytes,
    point_cloud_name: str,
    yaml_bytes: bytes,
    yaml_name: str,
) -> AppState:
    state: AppState = {
        "question": question,
        "point_cloud_bytes": point_cloud_bytes,
        "point_cloud_name": point_cloud_name,
        "yaml_bytes": yaml_bytes,
        "yaml_name": yaml_name,
    }

    app_graph = build_app_graph()
    if app_graph is not None:
        return app_graph.invoke(state)

    state = perception_node(state)
    state = rag_node(state)
    state = mcp_node(state)
    state = coordinator_node(state)
    return state
