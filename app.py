from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from coordinator import run_pipeline
from sample_assets import get_sample_scene, list_sample_scene_names


def make_point_cloud_figure(plot_data: dict, centroid_data: dict | None = None) -> go.Figure:
    marker_kwargs = {"size": 2}
    if "color" in plot_data:
        marker_kwargs["color"] = plot_data["color"]
    traces = [
        go.Scatter3d(
            x=plot_data["x"],
            y=plot_data["y"],
            z=plot_data["z"],
            mode="markers",
            marker=marker_kwargs,
            name="Point cloud",
        )
    ]
    if centroid_data and centroid_data.get("x"):
        traces.append(
            go.Scatter3d(
                x=centroid_data["x"],
                y=centroid_data["y"],
                z=centroid_data["z"],
                mode="markers+text",
                text=centroid_data["text"],
                hovertext=centroid_data.get("hover_text"),
                hoverinfo="text",
                textposition="top center",
                marker={"size": 6, "color": "red", "symbol": "diamond"},
                name="Obstacle labels",
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        title="Point cloud preview",
    )
    return fig


def risk_badge(risk: str) -> str:
    return risk


def describe_point_cloud_stats(stats: dict) -> str:
    num_points = stats.get("num_points")
    num_obstacles = stats.get("num_obstacles")
    extractor = stats.get("extractor_used", "unknown")
    return (
        f"The uploaded point cloud contains {num_points} points. "
        f"The perception stage extracted {num_obstacles} obstacle cluster(s) using the {extractor} pipeline."
    )


def describe_robot_state(mcp_output: dict) -> str:
    pose = mcp_output.get("pose", {})
    speed = float(mcp_output.get("speed_mps", 0.0))
    actor_count = len(mcp_output.get("nearby_actors", []))
    x = pose.get("x", 0.0)
    y = pose.get("y", 0.0)
    yaw = pose.get("yaw_deg", 0.0)

    if speed < 0.1:
        speed_text = "currently stationary"
    elif speed < 1.0:
        speed_text = f"moving slowly at {speed:.1f} m/s"
    else:
        speed_text = f"moving at {speed:.1f} m/s"

    return (
        f"The ego robot is {speed_text}. Its LiDAR pose is approximately x={x:.1f}, y={y:.1f}, yaw={yaw:.1f} deg. "
        f"Scenario metadata indicates {actor_count} nearby actor(s) in the environment."
    )


def describe_retrieval(docs: list[dict]) -> str:
    if not docs:
        return "No policy documents were retrieved."
    titles = [doc.get("title", "untitled") for doc in docs]
    titles_text = ", ".join(titles)
    return (
        "The recommendation is grounded in retrieved operational knowledge, including "
        f"{titles_text}."
    )


def actor_table_rows(mcp_output: dict) -> list[dict]:
    rows = []
    for actor in mcp_output.get("nearby_actors", [])[:8]:
        rows.append(
            {
                "ID": actor.get("id", "?"),
                "Type": actor.get("type", "unknown"),
                "Distance (m)": round(float(actor.get("distance_m", 0.0)), 2),
                "Speed (m/s)": round(float(actor.get("speed_mps", 0.0)), 2),
            }
        )
    return rows


st.set_page_config(page_title="Agentic Robot Scene Risk Analyzer", layout="wide")
st.title("Agentic Robot Scene Risk Analyzer")
st.text(
    "This demo showcases an agentic robotics scene analyzer. It combines point-cloud interpretation, retrieved safety knowledge, scenario metadata, and coordinated reasoning."
)

with st.sidebar:
    st.subheader("Analysis setup")
    question = st.text_input(
        "Question",
        value="Is the path blocked, and what should the robot do?",
    )
    input_mode = st.radio(
        "Scene source",
        options=["Use bundled sample", "Upload my own files"],
        index=0,
    )

    sample_scene_name = None
    if input_mode == "Use bundled sample":
        sample_scene_name = st.selectbox(
            "Sample scene",
            options=list_sample_scene_names(),
        )
        # st.caption(get_sample_scene(sample_scene_name)["description"])

    st.markdown(
        "Use a bundled sample scene or upload a `.pcd` or `.ply` file plus the matching `.yaml` metadata file. "
        "The app visualizes the scene, extracts obstacle descriptions, retrieves safety context, and recommends a navigation action."
    )

uploaded_file = None
yaml_file = None
if input_mode == "Upload my own files":
    uploaded_file = st.file_uploader("Upload a point cloud file", type=["pcd", "ply"])
    yaml_file = st.file_uploader("Upload matching YAML metadata", type=["yaml", "yml"])

ready_to_run = sample_scene_name is not None if input_mode == "Use bundled sample" else (
    uploaded_file is not None and yaml_file is not None
)
run = st.button(
    "Run analysis",
    type="primary",
    disabled=not ready_to_run,
)

if not ready_to_run:
    st.info("Choose a bundled sample scene or upload a point cloud file and matching YAML metadata to start.")

if run and ready_to_run:
    if input_mode == "Use bundled sample":
        selected_scene = get_sample_scene(sample_scene_name)
        file_bytes = selected_scene["point_cloud_bytes"]
        yaml_bytes = selected_scene["yaml_bytes"]
        point_cloud_name = selected_scene["point_cloud_name"]
        yaml_name = selected_scene["yaml_name"]
        st.caption(f"Running bundled sample: {sample_scene_name}")
    else:
        file_bytes = uploaded_file.read()
        yaml_bytes = yaml_file.read()
        point_cloud_name = uploaded_file.name
        yaml_name = yaml_file.name

    final_state = run_pipeline(
        question=question,
        point_cloud_bytes=file_bytes,
        point_cloud_name=point_cloud_name,
        yaml_bytes=yaml_bytes,
        yaml_name=yaml_name,
    )

    perception_output = final_state["perception_output"]
    mcp_output = final_state["mcp_output"]
    final_output = final_state["final_output"]
    rag_output = final_state["rag_output"]

    st.subheader("Executive summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Risk level", risk_badge(final_output.get("risk_level", "Unknown")))
    m2.metric(
        "Detected obstacle clusters",
        perception_output.get("point_cloud_stats", {}).get("num_obstacles", 0),
    )
    m3.metric(
        "Nearby actors in metadata",
        len(mcp_output.get("nearby_actors", [])),
    )

    st.markdown("#### Recommended action")
    st.write(final_output.get("recommended_action", "No recommendation available."))

    st.markdown("#### Why")
    st.write(final_output.get("scene_assessment", "No assessment available."))

    overview_col, viz_col = st.columns([1, 1.4])

    with overview_col:
        st.subheader("Scene analysis")
        st.write(describe_point_cloud_stats(perception_output.get("point_cloud_stats", {})))
        st.write(perception_output.get("scene_summary", ""))
        st.write(describe_robot_state(mcp_output))
        st.write(describe_retrieval(rag_output))

        semantic_text = perception_output.get("semantic_text", perception_output.get("scene_text", ""))
        if semantic_text:
            preview = semantic_text.split("\n")[0:2]
            st.markdown("#### Interpreted obstacle descriptions")
            st.text("\n".join(preview))
            with st.expander("Show full interpretation"):
                st.text(semantic_text)
                
        evidence = final_output.get("evidence", [])
        if evidence:
            st.markdown("#### Key supporting evidence")
            for item in evidence[:4]:
                st.write(f"- {item}")

    with viz_col:
        st.subheader("3D scene visualization")
        st.plotly_chart(
            make_point_cloud_figure(
                perception_output["plot_data"],
                perception_output.get("centroid_data"),
            ),
            use_container_width=True,
        )

    st.subheader("Technical details")
    tech_tab1, tech_tab2, tech_tab3 = st.tabs([
        "Robot state",
        "Retrieved context",
        "Raw outputs",
    ])

    with tech_tab1:
        st.write(describe_robot_state(mcp_output))
        actor_rows = actor_table_rows(mcp_output)
        if actor_rows:
            st.dataframe(actor_rows, use_container_width=True, hide_index=True)
        else:
            st.write("No nearby actors were found in the selected metadata.")
        with st.expander("Show raw metadata-derived state"):
            st.json(mcp_output)

    with tech_tab2:
        st.write(describe_retrieval(rag_output))
        for doc in rag_output:
            with st.expander(doc["title"]):
                st.write(doc["content"])

    with tech_tab3:
        with st.expander("Show structured perception output"):
            st.json(perception_output["scene"])
        with st.expander("Show point-cloud statistics"):
            st.json(perception_output.get("point_cloud_stats", {}))
        with st.expander("Show raw geometry-derived text"):
            st.text(perception_output.get("raw_scene_text", ""))
        with st.expander("Show obstacle descriptions (structured)"):
            st.json(perception_output.get("obstacle_descriptions", []))
