import numpy as np

# Workspace limits - conservative to ensure robot can reach all positions
WORKSPACE_RADIUS_RANGE = (0.25, 0.5)
WORKSPACE_ANGLE_RANGE_CUBE = (-np.deg2rad(80), np.deg2rad(80))  # ±80 degrees for cube
WORKSPACE_ANGLE_RANGE_CUP = (-np.deg2rad(65), np.deg2rad(65))   # ±65 degrees for cup
MIN_OBJECT_SEPARATION = 0.06
CUP_CUBE_MIN_DISTANCE = 0.10

_FORWARD_AXIS = np.array([0.0, -1.0], dtype=float)
_FORWARD_AXIS /= np.linalg.norm(_FORWARD_AXIS)
_RIGHT_AXIS = np.array([_FORWARD_AXIS[1], -_FORWARD_AXIS[0]], dtype=float)
_RIGHT_AXIS /= np.linalg.norm(_RIGHT_AXIS)


def sample_workspace_xy(rng, existing=None, min_separation=None, max_attempts=32, angle_range=None):
    if existing is None:
        existing = []
    if min_separation is None:
        min_separation = MIN_OBJECT_SEPARATION
    if angle_range is None:
        angle_range = WORKSPACE_ANGLE_RANGE_CUBE  # Default to cube range

    candidate = None
    for _ in range(max_attempts):
        radius = rng.uniform(*WORKSPACE_RADIUS_RANGE)
        angle = rng.uniform(*angle_range)
        forward_component = radius * np.cos(angle)
        right_component = radius * np.sin(angle)
        candidate = forward_component * _FORWARD_AXIS + right_component * _RIGHT_AXIS
        if all(np.linalg.norm(candidate - other) >= min_separation for other in existing):
            return candidate

    if candidate is not None:
        return candidate

    fallback = _FORWARD_AXIS * WORKSPACE_RADIUS_RANGE[0]
    if existing:
        for shift_sign in (1, -1):
            offset = fallback + shift_sign * min_separation * _RIGHT_AXIS
            if all(np.linalg.norm(offset - other) >= min_separation for other in existing):
                return offset
    return fallback
