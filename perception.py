from __future__ import annotations

from integrated_pointcloud_tool import extract_scene_from_bytes


def extract_scene_from_upload(file_bytes: bytes, filename: str):
    return extract_scene_from_bytes(file_bytes, filename)
