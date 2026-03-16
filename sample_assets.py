from __future__ import annotations

from pathlib import Path
from typing import Dict, List


SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"

SAMPLE_SCENES: Dict[str, Dict[str, str]] = {
    "Scene 1": {
        "point_cloud": "000060.pcd",
        "yaml": "000060.yaml",
        "description": "Scene 1.",
    },
    "Scene 2": {
        "point_cloud": "000062.pcd",
        "yaml": "000062.yaml",
        "description": "Scene 2.",
    },
}


def list_sample_scene_names() -> List[str]:
    return list(SAMPLE_SCENES.keys())


def get_sample_scene(scene_name: str) -> Dict[str, object]:
    if scene_name not in SAMPLE_SCENES:
        raise KeyError(f"Unknown sample scene: {scene_name}")

    scene = SAMPLE_SCENES[scene_name]
    point_cloud_path = SAMPLE_DATA_DIR / scene["point_cloud"]
    yaml_path = SAMPLE_DATA_DIR / scene["yaml"]

    return {
        "name": scene_name,
        "description": scene["description"],
        "point_cloud_name": point_cloud_path.name,
        "point_cloud_bytes": point_cloud_path.read_bytes(),
        "yaml_name": yaml_path.name,
        "yaml_bytes": yaml_path.read_bytes(),
    }
