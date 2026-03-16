from __future__ import annotations

import io
from typing import Any, Dict, List

import yaml


def _load_yaml_bytes(yaml_bytes: bytes) -> Dict[str, Any]:
    return yaml.safe_load(io.BytesIO(yaml_bytes)) or {}


def _pose_from_list(values: List[float] | None) -> Dict[str, float]:
    values = values or []
    if len(values) < 6:
        return {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll_deg": 0.0,
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
        }

    return {
        "x": float(values[0]),
        "y": float(values[1]),
        "z": float(values[2]),
        "roll_deg": float(values[3]),
        "yaw_deg": float(values[4]),
        "pitch_deg": float(values[5]),
    }


def get_robot_state_from_yaml(yaml_bytes: bytes) -> Dict[str, Any]:
    data = _load_yaml_bytes(yaml_bytes)

    return {
        "pose": _pose_from_list(data.get("lidar_pose")),
        "true_ego_pos": _pose_from_list(data.get("true_ego_pos")),
        "predicted_ego_pos": _pose_from_list(data.get("predicted_ego_pos")),
        "speed_mps": float(data.get("ego_speed", 0.0)),
        "planner_status": "unknown",
        "robot_width_m": 0.55,
        "localization_confidence": 1.0,
        "scenario_metadata": {
            "yaml_loaded": True,
            "num_actors": len(data.get("vehicles", {})),
        },
    }


def get_nearby_actors_from_yaml(yaml_bytes: bytes, max_results: int = 10) -> Dict[str, Any]:
    data = _load_yaml_bytes(yaml_bytes)
    vehicles = data.get("vehicles", {}) or {}
    lidar_pose = data.get("lidar_pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ego_x = float(lidar_pose[0]) if len(lidar_pose) > 0 else 0.0
    ego_y = float(lidar_pose[1]) if len(lidar_pose) > 1 else 0.0

    nearby = []
    for actor_id, actor_data in vehicles.items():
        location = actor_data.get("location", [0.0, 0.0, 0.0])
        vx = float(location[0]) if len(location) > 0 else 0.0
        vy = float(location[1]) if len(location) > 1 else 0.0
        vz = float(location[2]) if len(location) > 2 else 0.0

        dx = vx - ego_x
        dy = vy - ego_y
        distance = (dx**2 + dy**2) ** 0.5

        nearby.append(
            {
                "id": str(actor_id),
                "type": actor_data.get("bp_id", "unknown"),
                "x": vx,
                "y": vy,
                "z": vz,
                "distance_m": distance,
                "speed_mps": float(actor_data.get("speed", 0.0)),
                "extent": actor_data.get("extent", []),
                "yaw_deg": float(actor_data.get("angle", [0.0, 0.0, 0.0])[1])
                if isinstance(actor_data.get("angle"), list) and len(actor_data.get("angle", [])) > 1
                else 0.0,
            }
        )

    nearby.sort(key=lambda item: item["distance_m"])
    return {"nearby_actors": nearby[:max_results]}
