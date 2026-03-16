from typing import Any, Dict, List, TypedDict


class AppState(TypedDict, total=False):
    question: str
    point_cloud_name: str
    point_cloud_bytes: bytes
    yaml_name: str
    yaml_bytes: bytes
    perception_output: Dict[str, Any]
    rag_output: List[Dict[str, str]]
    mcp_output: Dict[str, Any]
    final_output: Dict[str, Any]
