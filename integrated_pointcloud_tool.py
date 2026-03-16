from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except Exception:
    OPEN3D_AVAILABLE = False


def _require_open3d() -> None:
    if not OPEN3D_AVAILABLE:
        raise RuntimeError("open3d is not installed. Install it to load and analyze point clouds.")


def load_point_cloud_from_bytes(file_bytes: bytes, suffix: str):
    _require_open3d()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        pcd = o3d.io.read_point_cloud(tmp_path)
        points = np.asarray(pcd.points)
        if len(points) == 0:
            raise ValueError("The uploaded point cloud is empty.")

        # Preserve the orientation used in the user's tool.
        points[:, 1] = -points[:, 1]
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def segment_ground(pcd, distance_threshold: float = 0.2, ransac_n: int = 3, num_iterations: int = 200):
    _require_open3d()
    if len(pcd.points) < max(ransac_n + 1, 4):
        raise ValueError(f"Not enough points to segment ground: found {len(pcd.points)} points.")

    try:
        _, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
    except RuntimeError as exc:
        if "ransac_n" in str(exc):
            raise ValueError("Not enough valid points for RANSAC plane segmentation.") from exc
        raise

    ground = pcd.select_by_index(inliers)
    objects = pcd.select_by_index(inliers, invert=True)
    return ground, objects


def cluster_obstacles(objects, eps: float = 0.5, min_points: int = 10, ego_distance_threshold: float = 2.1):
    _require_open3d()
    if len(objects.points) == 0:
        return [], np.array([], dtype=int)

    labels = np.array(objects.cluster_dbscan(eps=eps, min_points=min_points))
    clusters = []
    kept_cluster_ids = []

    for i in np.unique(labels):
        if i == -1:
            continue
        cluster = objects.select_by_index(np.where(labels == i)[0])
        points = np.asarray(cluster.points)
        if len(points) == 0:
            continue
        centroid = points.mean(axis=0)
        if np.linalg.norm(centroid[:2]) > ego_distance_threshold:
            clusters.append(cluster)
            kept_cluster_ids.append(i)

    remapped_labels = np.full_like(labels, fill_value=-1)
    for new_id, old_id in enumerate(kept_cluster_ids):
        remapped_labels[labels == old_id] = new_id
    return clusters, remapped_labels


def describe_obstacles(clusters) -> List[Dict[str, Any]]:
    obstacle_info: List[Dict[str, Any]] = []
    for i, cluster in enumerate(clusters):
        points = np.asarray(cluster.points)
        centroid = points.mean(axis=0)
        pmin = points.min(axis=0)
        pmax = points.max(axis=0)
        size = pmax - pmin
        volume = float(np.prod(size)) if np.prod(size) != 0 else 0.0
        density = float(len(points) / volume) if volume > 0 else -1.0
        obstacle_info.append(
            {
                "id": i,
                "centroid_relative_to_robot": centroid.tolist(),
                "bounding_box": {
                    "min": pmin.tolist(),
                    "max": pmax.tolist(),
                    "size": size.tolist(),
                },
                "num_points": int(len(points)),
                "density": density,
            }
        )
    return obstacle_info


def obstacles_to_text(obstacle_info: List[Dict[str, Any]]) -> str:
    text_descriptions: List[str] = []
    for obs in obstacle_info:
        c = obs["centroid_relative_to_robot"]
        s = obs["bounding_box"]["size"]
        d = obs["density"]
        distance = math.trunc(float(np.linalg.norm(c)))
        text_descriptions.append(
            f"Obstacle {obs['id']}: at x={c[0]:.2f}m, y={c[1]:.2f}m, z={c[2]:.2f}m; "
            f"Distance from ego vehicle: {distance} m; "
            f"Approximate size x={s[0]:.2f}m, y={s[1]:.2f}m, z={s[2]:.2f}m; "
            f"Density={d:.2f}"
        )
    return "\n".join(text_descriptions)


# ---- semantic interpretation helpers ----

def _region_from_xy(x: float, y: float, lateral_thresh: float = 1.5) -> str:
    if x >= 0:
        if abs(y) < lateral_thresh:
            return "front"
        return "front-left" if y > 0 else "front-right"
    if abs(y) < lateral_thresh:
        return "behind"
    return "rear-left" if y > 0 else "rear-right"


def _distance_band(distance: float) -> str:
    if distance < 2.0:
        return "very close"
    if distance < 5.0:
        return "near"
    if distance < 10.0:
        return "mid-range"
    return "far"


def _size_category(size: List[float]) -> str:
    sx, sy, sz = [float(v) for v in size]
    footprint = sx * sy
    if footprint < 0.5 and sz < 1.0:
        return "small"
    if footprint < 2.0 and sz < 2.0:
        return "medium"
    return "large"


def _density_category(density: float) -> str:
    if density < 0:
        return "unknown-density"
    if density < 5:
        return "sparse"
    if density < 20:
        return "moderately dense"
    return "dense"


def _risk_hint(region: str, distance: float, size_cat: str, density_cat: str) -> str:
    if region == "front" and distance < 3.0:
        if size_cat == "large" or density_cat == "dense":
            return "likely blocking the forward path"
        return "requires caution"
    if region.startswith("front") and distance < 5.0:
        return "may affect near-term navigation"
    return "unlikely to immediately block navigation"


def _semantic_obstacle_record(obs: Dict[str, Any]) -> Dict[str, Any]:
    c = obs["centroid_relative_to_robot"]
    size = obs["bounding_box"]["size"]
    density = float(obs["density"])
    distance = float(np.linalg.norm(c))
    region = _region_from_xy(float(c[0]), float(c[1]))
    size_cat = _size_category(size)
    density_cat = _density_category(density)
    risk_hint = _risk_hint(region, distance, size_cat, density_cat)
    return {
        **obs,
        "semantic": {
            "region": region,
            "distance_m": round(distance, 2),
            "distance_band": _distance_band(distance),
            "size_category": size_cat,
            "density_category": density_cat,
            "risk_hint": risk_hint,
        },
    }


def obstacles_to_semantic_text(obstacle_info: List[Dict[str, Any]]) -> str:
    if not obstacle_info:
        return "No significant obstacle clusters were detected."

    descriptions: List[str] = []
    for obs in obstacle_info:
        semantic = obs.get("semantic") or _semantic_obstacle_record(obs)["semantic"]
        descriptions.append(
            f"Obstacle {obs['id']}: {semantic['size_category']} {semantic['density_category']} obstacle cluster in the "
            f"{semantic['region']}, {semantic['distance_band']} from the robot at about {semantic['distance_m']:.1f} m, "
            f"{semantic['risk_hint']}."
        )
    return "\n".join(descriptions)


def summarize_scene_semantically(obstacle_info: List[Dict[str, Any]]) -> str:
    if not obstacle_info:
        return "The scene appears mostly clear, with no significant obstacle clusters detected."

    semantic_records = [obs.get("semantic") or _semantic_obstacle_record(obs)["semantic"] for obs in obstacle_info]
    frontal = [s for s in semantic_records if s["region"].startswith("front")]
    close_frontal = [s for s in frontal if s["distance_m"] < 3.0]
    left_count = sum(1 for s in semantic_records if "left" in s["region"])
    right_count = sum(1 for s in semantic_records if "right" in s["region"])
    blocking_frontal = [s for s in frontal if "blocking" in s["risk_hint"]]

    summary_parts = [
        f"The scene contains {len(obstacle_info)} detected obstacle clusters.",
        f"{len(frontal)} cluster(s) are in the forward field of view.",
    ]
    if close_frontal:
        summary_parts.append(
            f"{len(close_frontal)} frontal cluster(s) are within 3 meters, indicating possible near-term blockage."
        )
    if blocking_frontal:
        summary_parts.append("At least one frontal cluster appears likely to block the robot's direct path.")
    if left_count < right_count:
        summary_parts.append("The left side appears less cluttered than the right.")
    elif right_count < left_count:
        summary_parts.append("The right side appears less cluttered than the left.")
    else:
        summary_parts.append("Obstacle clutter is roughly balanced laterally.")
    return " ".join(summary_parts)


# ---- scene conversion ----

def _compute_free_space(points: np.ndarray) -> Dict[str, float]:
    if len(points) == 0:
        return {"front": 1.0, "left": 1.0, "right": 1.0}

    x = points[:, 0]
    y = points[:, 1]
    near_mask = np.linalg.norm(points[:, :2], axis=1) <= 8.0
    if not np.any(near_mask):
        near_mask = np.ones(len(points), dtype=bool)

    near_x = x[near_mask]
    near_y = y[near_mask]
    front_occ = np.mean((near_x > 0.5) & (np.abs(near_y) < 1.5))
    left_occ = np.mean((near_x > 0.2) & (near_y > 0.8))
    right_occ = np.mean((near_x > 0.2) & (near_y < -0.8))

    return {
        "front": round(float(max(0.0, 1.0 - min(1.0, front_occ * 8.0))), 2),
        "left": round(float(max(0.0, 1.0 - min(1.0, left_occ * 8.0))), 2),
        "right": round(float(max(0.0, 1.0 - min(1.0, right_occ * 8.0))), 2),
    }


def obstacle_info_to_scene(obstacle_info: List[Dict[str, Any]], object_points: np.ndarray) -> Dict[str, Any]:
    objects: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    risk_flags: List[str] = []

    for obs in obstacle_info:
        semantic = obs.get("semantic") or _semantic_obstacle_record(obs)["semantic"]
        size = obs["bounding_box"]["size"]
        objects.append(
            {
                "id": f"obj_{obs['id']}",
                "type": "unknown_obstacle",
                "distance_m": semantic["distance_m"],
                "region": semantic["region"],
                "motion": "unknown",
                "num_points": obs["num_points"],
                "density": round(float(obs["density"]), 2),
                "size_category": semantic["size_category"],
                "density_category": semantic["density_category"],
                "risk_hint": semantic["risk_hint"],
            }
        )
        groups.append(
            {
                "group_id": f"g{obs['id']}",
                "members": 1,
                "center_distance_m": semantic["distance_m"],
                "span_m": round(float(max(size[0], size[1])), 2),
                "region": semantic["region"],
                "semantic_label": f"{semantic['size_category']} {semantic['region']} cluster",
            }
        )

    free_space = _compute_free_space(object_points)
    if free_space["front"] < 0.40:
        risk_flags.append("forward_corridor_partially_blocked")
    if free_space["front"] < 0.25:
        risk_flags.append("forward_path_highly_obstructed")
    if len(objects) >= 4:
        risk_flags.append("dense_obstacle_scene")

    if any(obj["region"] == "front" and obj["distance_m"] < 3.0 for obj in objects):
        risk_flags.append("close_frontal_obstacle")

    return {
        "objects": objects,
        "group_summary": groups,
        "free_space": free_space,
        "risk_flags": risk_flags,
    }


# ---- visualization helpers ----

def _downsample(points: np.ndarray, colors: np.ndarray, max_points: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, colors
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[idx], colors[idx]


def build_colored_plot_data(ground, objects, labels: np.ndarray) -> Dict[str, Any]:
    ground_points = np.asarray(ground.points)
    object_points = np.asarray(objects.points)

    ground_colors = np.tile(np.array([[0, 0, 255]], dtype=np.uint8), (len(ground_points), 1))

    rng = np.random.default_rng(42)
    object_colors = np.tile(np.array([[0, 200, 0]], dtype=np.uint8), (len(object_points), 1))
    if len(labels) == len(object_points):
        unique_labels = [l for l in np.unique(labels) if l >= 0]
        palette = {label: rng.integers(30, 255, size=3, dtype=np.uint8) for label in unique_labels}
        for label, color in palette.items():
            object_colors[labels == label] = color
        object_colors[labels < 0] = np.array([160, 160, 160], dtype=np.uint8)

    all_points = np.vstack([ground_points, object_points]) if len(ground_points) else object_points
    all_colors = np.vstack([ground_colors, object_colors]) if len(ground_points) else object_colors
    all_points, all_colors = _downsample(all_points, all_colors)

    color_strings = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in all_colors]
    return {
        "x": all_points[:, 0].tolist(),
        "y": all_points[:, 1].tolist(),
        "z": all_points[:, 2].tolist(),
        "color": color_strings,
    }


def centroid_plot_data(obstacle_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    texts: List[str] = []
    hover_texts: List[str] = []
    for obs in obstacle_info:
        c = obs["centroid_relative_to_robot"]
        semantic = obs.get("semantic") or _semantic_obstacle_record(obs)["semantic"]
        xs.append(float(c[0]))
        ys.append(float(c[1]))
        zs.append(float(c[2]))
        # texts.append(f"{obs['id']} | {semantic['region']} | {semantic['distance_band']}")
        texts.append(f"{obs['id']}")
        hover_texts.append(
            f"Obstacle {obs['id']}<br>Region: {semantic['region']}<br>Distance: {semantic['distance_m']} m"
            f"<br>Size: {semantic['size_category']}<br>Density: {semantic['density_category']}"
            f"<br>Hint: {semantic['risk_hint']}"
        )
    return {"x": xs, "y": ys, "z": zs, "text": texts, "hover_text": hover_texts}


def extract_scene_from_bytes(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    pcd = load_point_cloud_from_bytes(file_bytes, suffix)
    ground, objects = segment_ground(pcd)
    clusters, labels = cluster_obstacles(objects)
    obstacle_info = describe_obstacles(clusters)
    semantic_obstacles = [_semantic_obstacle_record(obs) for obs in obstacle_info]

    raw_text = obstacles_to_text(semantic_obstacles)
    semantic_text = obstacles_to_semantic_text(semantic_obstacles)
    scene_summary = summarize_scene_semantically(semantic_obstacles)

    object_points = np.asarray(objects.points) if len(objects.points) else np.empty((0, 3))
    scene = obstacle_info_to_scene(semantic_obstacles, object_points)
    plot_data = build_colored_plot_data(ground, objects, labels)
    centroid_data = centroid_plot_data(semantic_obstacles)

    points = np.asarray(pcd.points)
    combined_scene_text = f"{scene_summary}\n\n{semantic_text}" if semantic_text else scene_summary
    return {
        "scene": scene,
        "scene_text": combined_scene_text,
        "scene_summary": scene_summary,
        "semantic_text": semantic_text,
        "raw_scene_text": raw_text if raw_text else "No raw obstacle descriptions could be extracted.",
        "plot_data": plot_data,
        "centroid_data": centroid_data,
        "obstacle_descriptions": semantic_obstacles,
        "point_cloud_stats": {
            "num_points": int(len(points)),
            "num_obstacles": int(len(semantic_obstacles)),
            "min_xyz": np.min(points, axis=0).round(3).tolist(),
            "max_xyz": np.max(points, axis=0).round(3).tolist(),
            "extractor_used": "integrated_interpret_pointcloud_tool_semantic",
            "source_file": filename,
        },
    }
