import numpy as np

WORKSPACE_RADIUS_RANGE = (0.35, 0.65)
WORKSPACE_ANGLE_RANGE = (-np.pi / 2, np.pi / 2)
MIN_OBJECT_SEPARATION = 0.18
CUP_CUBE_MIN_DISTANCE = 0.28

_FORWARD_AXIS = np.array([0.0, -1.0], dtype=float)
_FORWARD_AXIS /= np.linalg.norm(_FORWARD_AXIS)
_RIGHT_AXIS = np.array([_FORWARD_AXIS[1], -_FORWARD_AXIS[0]], dtype=float)
_RIGHT_AXIS /= np.linalg.norm(_RIGHT_AXIS)


def sample_workspace_xy(rng, existing=None, min_separation=None, max_attempts=32):
    if existing is None:
        existing = []
    if min_separation is None:
        min_separation = MIN_OBJECT_SEPARATION

    candidate = None
    for _ in range(max_attempts):
        radius = rng.uniform(*WORKSPACE_RADIUS_RANGE)
        angle = rng.uniform(*WORKSPACE_ANGLE_RANGE)
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
