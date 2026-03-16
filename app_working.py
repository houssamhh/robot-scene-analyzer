import os
import json
from typing import Any, Dict, List, TypedDict

import streamlit as st

# Optional imports: the app still runs in fallback mode if these are missing.
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False


# =========================
# Demo data and utilities
# =========================
DEFAULT_SCENES = {
    "Narrow corridor with front obstacles": {
        "objects": [
            {"id": "obj_1", "type": "box", "distance_m": 2.1, "region": "front-left", "motion": "static"},
            {"id": "obj_2", "type": "cart", "distance_m": 2.4, "region": "front", "motion": "static"},
            {"id": "obj_3", "type": "person", "distance_m": 2.8, "region": "front-right", "motion": "slow"}
        ],
        "group_summary": [
            {"group_id": "g1", "members": 3, "center_distance_m": 2.43, "span_m": 1.7, "region": "front"}
        ],
        "free_space": {"front": 0.35, "left": 0.52, "right": 0.76},
        "risk_flags": ["forward_corridor_partially_blocked"]
    },
    "Mostly open space": {
        "objects": [
            {"id": "obj_1", "type": "pole", "distance_m": 4.5, "region": "front-left", "motion": "static"},
            {"id": "obj_2", "type": "chair", "distance_m": 5.2, "region": "right", "motion": "static"}
        ],
        "group_summary": [],
        "free_space": {"front": 0.88, "left": 0.81, "right": 0.79},
        "risk_flags": []
    },
    "Dense frontal clutter": {
        "objects": [
            {"id": "obj_1", "type": "box", "distance_m": 1.9, "region": "front-left", "motion": "static"},
            {"id": "obj_2", "type": "box", "distance_m": 2.0, "region": "front", "motion": "static"},
            {"id": "obj_3", "type": "box", "distance_m": 2.2, "region": "front-right", "motion": "static"},
            {"id": "obj_4", "type": "person", "distance_m": 2.4, "region": "front", "motion": "slow"}
        ],
        "group_summary": [
            {"group_id": "g1", "members": 4, "center_distance_m": 2.12, "span_m": 1.4, "region": "front"}
        ],
        "free_space": {"front": 0.18, "left": 0.42, "right": 0.39},
        "risk_flags": ["forward_path_highly_obstructed", "dynamic_object_present"]
    }
}

RAG_DOCS = [
    {
        "title": "navigation_rules.md",
        "content": "If forward free-space ratio falls below 0.40, direct traversal is considered unsafe unless the robot footprint is exceptionally small. If one side corridor has free-space ratio above 0.70, bypass by replanning through that side is usually preferred over stopping."
    },
    {
        "title": "safety_thresholds.md",
        "content": "Robots must reduce speed when dynamic obstacles are within 3.0 meters in the frontal region. If frontal clutter includes a moving person, conservative slowdown is required even when a side bypass exists."
    },
    {
        "title": "planner_behavior_notes.md",
        "content": "The local planner performs poorly in narrow frontal clutter when obstacle groups span more than 1.2 meters across the corridor. Right-side bypass is preferred when right free-space exceeds left free-space by at least 0.15."
    },
    {
        "title": "platform_limits.md",
        "content": "Nominal robot width is 0.55 meters. Safe passage generally requires at least robot width plus 0.20 meters clearance. High speed should be avoided when localization confidence is below 0.80."
    },
    {
        "title": "known_scene_patterns.md",
        "content": "Dense static frontal clusters at 1.5 to 3.0 meters often indicate temporary blockage rather than full dead-end conditions. In such scenes, the recommended action is often slow down and replan rather than immediate stop, provided lateral free space is available."
    }
]


def point_cloud_to_structured_text(scene_json: Dict[str, Any]) -> Dict[str, Any]:
    """Demo perception tool.
    In a real project this would parse PCD/PLY data and extract obstacles.
    Here it converts scene JSON to a structured summary plus text.
    """
    objects = scene_json.get("objects", [])
    group_summary = scene_json.get("group_summary", [])
    free_space = scene_json.get("free_space", {})
    risk_flags = scene_json.get("risk_flags", [])

    text_lines = []
    if objects:
        text_lines.append(f"Detected {len(objects)} objects.")
        for obj in objects:
            text_lines.append(
                f"{obj['type']} at {obj['distance_m']}m in {obj['region']} moving {obj['motion']}."
            )
    if group_summary:
        for grp in group_summary:
            text_lines.append(
                f"Group {grp['group_id']} has {grp['members']} objects centered at {grp['center_distance_m']}m spanning {grp['span_m']}m in the {grp['region']} region."
            )
    if free_space:
        text_lines.append(
            f"Free space ratios - front: {free_space.get('front', 0)}, left: {free_space.get('left', 0)}, right: {free_space.get('right', 0)}."
        )
    if risk_flags:
        text_lines.append("Risk flags: " + ", ".join(risk_flags) + ".")

    return {
        "scene": scene_json,
        "scene_text": " ".join(text_lines),
    }


# =========================
# RAG layer
# =========================
def build_simple_rag_context(question: str, scene_text: str) -> List[Dict[str, str]]:
    """Fallback lexical retrieval so the demo works without external services."""
    query = f"{question} {scene_text}".lower()
    scored = []
    for doc in RAG_DOCS:
        content = doc["content"].lower()
        score = sum(1 for token in query.split() if token in content)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:3]]


def build_langchain_vectorstore():
    if not LANGCHAIN_AVAILABLE:
        return None
    try:
        embeddings = OpenAIEmbeddings()
        texts = [f"{d['title']}\n{d['content']}" for d in RAG_DOCS]
        metadatas = [{"title": d["title"]} for d in RAG_DOCS]
        return FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    except Exception:
        return None


# =========================
# MCP demo tools
# =========================
def get_robot_state() -> Dict[str, Any]:
    return {
        "pose": {"x": 12.4, "y": 5.8, "yaw_deg": 90},
        "speed_mps": 0.9,
        "robot_width_m": 0.55,
        "localization_confidence": 0.87,
        "goal": "reach docking station",
        "planner_status": "active"
    }


def get_nearby_robots() -> Dict[str, Any]:
    return {
        "nearby_robots": [
            {"id": "robot_2", "distance_m": 4.1, "bearing_deg": -35},
            {"id": "robot_3", "distance_m": 6.7, "bearing_deg": 22}
        ]
    }


# =========================
# Coordinator logic
# =========================
class AppState(TypedDict, total=False):
    question: str
    scene_input: Dict[str, Any]
    perception_output: Dict[str, Any]
    rag_output: List[Dict[str, str]]
    mcp_output: Dict[str, Any]
    final_output: Dict[str, Any]


def perception_node(state: AppState) -> AppState:
    return {**state, "perception_output": point_cloud_to_structured_text(state["scene_input"])}


def rag_node(state: AppState) -> AppState:
    question = state["question"]
    scene_text = state["perception_output"]["scene_text"]
    docs = build_simple_rag_context(question, scene_text)
    return {**state, "rag_output": docs}


def mcp_node(state: AppState) -> AppState:
    robot_state = get_robot_state()
    nearby = get_nearby_robots()
    return {**state, "mcp_output": {**robot_state, **nearby}}


def rule_based_decision(state: AppState) -> Dict[str, Any]:
    scene = state["perception_output"]["scene"]
    free_space = scene.get("free_space", {})
    risk_flags = scene.get("risk_flags", [])
    right = free_space.get("right", 0)
    left = free_space.get("left", 0)
    front = free_space.get("front", 0)
    dynamic_present = "dynamic_object_present" in risk_flags or any(
        o.get("motion") not in ("static", "none") for o in scene.get("objects", [])
    )

    if front < 0.25:
        risk = "High"
        recommendation = "Stop or perform conservative replanning before moving forward."
    elif front < 0.40 and max(left, right) > 0.70:
        risk = "Medium"
        recommendation = "Slow down and replan through the more open side corridor."
    elif front < 0.40:
        risk = "Medium"
        recommendation = "Reduce speed and request a new local plan."
    else:
        risk = "Low"
        recommendation = "Proceed with caution."

    if dynamic_present and risk != "High":
        recommendation = "Slow down due to dynamic obstacle presence, then re-evaluate or replan."

    evidence = [
        f"Front free-space ratio is {front}.",
        f"Left free-space ratio is {left}; right free-space ratio is {right}.",
    ]
    evidence.extend([f"Retrieved rule: {doc['content']}" for doc in state.get("rag_output", [])[:2]])
    evidence.append(f"Robot speed is {state['mcp_output']['speed_mps']} m/s with localization confidence {state['mcp_output']['localization_confidence']}.")

    return {
        "risk_level": risk,
        "scene_assessment": state["perception_output"]["scene_text"],
        "recommended_action": recommendation,
        "evidence": evidence,
    }


def coordinator_node(state: AppState) -> AppState:
    if LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0)
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a robotics decision-support coordinator. Return compact JSON with keys risk_level, scene_assessment, recommended_action, evidence. Base your answer only on the supplied perception summary, retrieved context, and robot state."
                ),
                (
                    "human",
                    "Question: {question}\n\nPerception summary: {scene_text}\n\nRetrieved context: {retrieved}\n\nRobot state: {robot_state}"
                ),
            ])
            chain = prompt | llm | StrOutputParser()
            raw = chain.invoke({
                "question": state["question"],
                "scene_text": state["perception_output"]["scene_text"],
                "retrieved": json.dumps(state.get("rag_output", []), indent=2),
                "robot_state": json.dumps(state.get("mcp_output", {}), indent=2),
            })
            final_output = json.loads(raw)
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


# =========================
# UI
# =========================
st.set_page_config(page_title="Agentic Robot Scene Risk Analyzer", layout="wide")
st.title("Agentic Robot Scene Risk Analyzer")
st.caption("A compact demo using a perception tool, a RAG layer, MCP-style robot state tools, and a coordinator.")

with st.sidebar:
    st.subheader("Demo setup")
    selected_scene_name = st.selectbox("Choose a demo scene", list(DEFAULT_SCENES.keys()))
    question = st.text_input(
        "Question",
        value="Is the path blocked, and what should the robot do?"
    )
    st.markdown("---")
    st.write("You can also edit the structured scene JSON below.")

scene_json = st.text_area(
    "Structured scene input",
    value=json.dumps(DEFAULT_SCENES[selected_scene_name], indent=2),
    height=360,
)

run = st.button("Run analysis", type="primary")

if run:
    try:
        scene_input = json.loads(scene_json)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    state: AppState = {
        "question": question,
        "scene_input": scene_input,
    }

    app_graph = build_app_graph()
    if app_graph is not None:
        final_state = app_graph.invoke(state)
    else:
        final_state = perception_node(state)
        final_state = rag_node(final_state)
        final_state = mcp_node(final_state)
        final_state = coordinator_node(final_state)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Perception output")
        st.json(final_state["perception_output"]["scene"])
        st.write(final_state["perception_output"]["scene_text"])

        st.subheader("Retrieved policy context")
        for doc in final_state["rag_output"]:
            with st.expander(doc["title"]):
                st.write(doc["content"])

    with col2:
        st.subheader("Robot state (MCP-style tools)")
        st.json(final_state["mcp_output"])

        st.subheader("Coordinator decision")
        st.json(final_state["final_output"])

st.markdown("---")
st.markdown("### How to run")
st.code("pip install streamlit langgraph langchain langchain-openai langchain-community faiss-cpu\nstreamlit run agentic_robot_scene_risk_analyzer_app.py")
