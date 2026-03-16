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
