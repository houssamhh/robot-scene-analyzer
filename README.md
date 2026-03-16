# Robot Scene Risk Analyzer

Robot Scene Risk Analyzer is a Streamlit demo for turning raw robot scene inputs into an explainable navigation recommendation.

It combines:
- point-cloud parsing and obstacle extraction,
- YAML-based robot and actor metadata,
- lightweight retrieval over safety and planner guidance,
- and a coordinator that produces a risk level, scene assessment, and recommended action.

The result is not just a visualization tool. It is a compact decision-support workflow for robotics scenarios.

## Why This Project Exists

A robot rarely makes decisions from one signal alone.

A useful navigation decision usually needs at least four inputs:
- geometry from the scene,
- state from the robot,
- context from operational knowledge,
- and a policy for converting those signals into action.

This project demonstrates that end-to-end loop in a way that is easy to run, inspect, and extend.

## What You Can Do

With the app, you can:
- run bundled sample scenes immediately, without uploading anything,
- upload your own `.pcd` or `.ply` point cloud and matching `.yaml` metadata,
- inspect a 3D scene visualization,
- review extracted obstacle clusters and semantic scene summaries,
- view metadata-derived robot state and nearby actors,
- inspect retrieved safety context,
- and generate a recommended motion decision with supporting evidence.

## Demo Experience

The Streamlit interface is organized around a practical robotics workflow:

1. Choose a bundled sample scene or upload your own files.
2. Ask a question such as `Is the path blocked, and what should the robot do?`
3. Run the pipeline.
4. Review the risk level, recommendation, evidence, and technical details.

Two sample scenes are included in `sample_data/` so the project is usable out of the box.

## Pipeline Overview

The application follows a simple multi-stage pipeline:

1. `Perception`
   Reads the point cloud, segments the ground, clusters obstacles, and converts geometry into scene-level summaries.

2. `RAG`
   Retrieves relevant navigation and safety snippets from a small in-repo knowledge base.

3. `MCP-style state tools`
   Extracts robot pose, speed, and nearby actors from the uploaded YAML metadata.

4. `Coordinator`
   Combines perception output, retrieved context, and robot state into a final recommendation.

When LangGraph and LangChain are available, the app can use those components. When they are not, the repo still works with fallback logic so the demo remains runnable.

## Project Structure

Key files:

- `app.py`: Streamlit UI and analysis flow.
- `coordinator.py`: Orchestrates perception, retrieval, state access, and final decision-making.
- `integrated_pointcloud_tool.py`: Point-cloud loading, segmentation, clustering, semantic labeling, and plotting data preparation.
- `perception.py`: Thin wrapper around the point-cloud extraction pipeline.
- `mcp_tools.py`: YAML parsing for robot state and nearby actors.
- `rag.py`: Simple retrieval layer plus optional LangChain vector store support.
- `sample_assets.py`: Registry and loader for bundled sample scenes.
- `sample_data/`: Built-in point cloud and YAML pairs.
- `demo_data.py`: In-repo safety and planner knowledge used by the retrieval layer.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the app

```bash
streamlit run app.py
```

### 3. Open the UI

In the sidebar:
- choose `Use bundled sample` for the fastest start, or
- choose `Upload my own files` to analyze your own scene.

## Input Requirements

### Point cloud

Supported formats:
- `.pcd`
- `.ply`

### Metadata

Supported formats:
- `.yaml`
- `.yml`

The YAML file is expected to contain fields such as:
- `lidar_pose`
- `true_ego_pos`
- `predicted_ego_pos`
- `ego_speed`
- `vehicles`

## Outputs

For each run, the app produces:
- a risk level such as `Low`, `Medium`, or `High`,
- a recommended action,
- a scene assessment,
- supporting evidence,
- a 3D point-cloud visualization,
- obstacle descriptions,
- robot-state details,
- and retrieved contextual documents.

## Extending the Project

### Bring your own point-cloud extractor

If you want to replace the built-in extraction logic, add a `user_scene_extractor.py` file in the project root with one of these interfaces:

```python
def extract_scene_from_point_cloud(file_path: str) -> dict:
    ...
```

or

```python
def extract_scene_from_point_cloud(file_bytes: bytes, filename: str) -> dict:
    ...
```

The extractor should return a dictionary shaped like:

```python
{
    "objects": [
        {
            "id": "obj_1",
            "type": "box",
            "distance_m": 2.3,
            "region": "front-left",
            "motion": "static",
        }
    ],
    "group_summary": [
        {
            "group_id": "g1",
            "members": 2,
            "center_distance_m": 2.4,
            "span_m": 1.1,
            "region": "front",
        }
    ],
    "free_space": {"front": 0.38, "left": 0.62, "right": 0.81},
    "risk_flags": ["forward_corridor_partially_blocked"],
}
```

### Swap out the decision policy

The final recommendation currently comes from:
- a rule-based coordinator by default,
- or an LLM-backed coordinator when the required LangChain/OpenAI setup is available.

If you want a stricter planning policy, `coordinator.py` is the place to change it.

## Optional LLM Support

The repo is usable without an API key.

If you do provide `OPENAI_API_KEY`, the coordinator can attempt an LLM-based decision path. If that path fails, it falls back to the rule-based logic.

## Dependencies

The project currently depends on:
- `streamlit`
- `plotly`
- `open3d`
- `numpy`
- `pyyaml`
- `langgraph`
- `langchain`
- `langchain-openai`
- `langchain-community`
- `faiss-cpu`

## Limitations

This is a demo-oriented analyzer, not a production autonomy stack.

Important limitations:
- the retrieval corpus is small and in-repo,
- the rule-based coordinator is intentionally simple,
- YAML parsing assumes a narrow metadata structure,
- and perception quality depends heavily on the input point cloud and `open3d` processing assumptions.

## Good Use Cases

This repo is a good fit for:
- robotics demos,
- hackathon prototypes,
- explainable autonomy experiments,
- point-cloud UI experiments,
- and teaching how multi-stage robot reasoning pipelines fit together.

## Summary

If you want a compact robotics demo that feels closer to a system than a chatbot, this project gives you:
- real point-cloud ingestion,
- interpretable scene summaries,
- metadata-aware reasoning,
- retrieval-grounded recommendations,
- and a UI that makes the full pipeline visible.
