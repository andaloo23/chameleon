#!/usr/bin/env python3
"""
Unit tests for the fingertip-to-face-zone box-distance formula.

Validates that compute_fingertip_obb_reach_reward produces correct distance
values for known fingertip positions relative to a cube at various orientations.
"""

import math
import torch
import pytest


def _box_distance_single(
    tip_world,       # [3] fingertip world position
    cube_pos,        # [3] cube center world position
    cube_quat_wxyz,  # [4] cube quaternion (w, x, y, z)
    use_x,           # bool: True if face normal is local X axis
    sign,            # +1 or -1: face side
    cube_half_size,  # float
    zone_margin,     # float
):
    """
    Compute box-distance for a single tip to a single face zone.
    Mirrors the formula in compute_fingertip_obb_reach_reward.
    """
    # Build rotation matrix from quaternion
    w, x, y, z = cube_quat_wxyz
    R = torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)    ],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)    ],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
    R_inv = R.T

    # Transform tip to cube-local space
    q = R_inv @ (tip_world - cube_pos)

    # Zone slab center in local space
    t = zone_margin
    face_offset = cube_half_size + 0.5 * t
    axis_local = torch.zeros(3)
    if use_x:
        axis_local[0] = 1.0
    else:
        axis_local[1] = 1.0
    center = sign * face_offset * axis_local

    # Half-extents
    h = torch.full((3,), cube_half_size)
    half_t = t / 2.0
    if use_x:
        h[0] = half_t
    else:
        h[1] = half_t

    # Box distance
    d = torch.linalg.norm(torch.clamp((q - center).abs() - h, min=0.0))
    return d.item(), q


class TestBoxDistance:
    """Tests for the box-distance formula used in fingertip OBB reward."""

    # Cube params: 3cm cube with 1.5cm zone margin
    HALF = 0.015
    MARGIN = 0.015

    def test_tip_at_zone_center_identity_quat(self):
        """Tip exactly at the center of the +X face zone -> d = 0."""
        cube_pos = torch.zeros(3)
        cube_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        # Zone center: x = half + margin/2 = 0.015 + 0.0075 = 0.0225
        tip = torch.tensor([0.0225, 0.0, 0.0])
        d, _ = _box_distance_single(tip, cube_pos, cube_quat, use_x=True, sign=1.0,
                                     cube_half_size=self.HALF, zone_margin=self.MARGIN)
        assert d == pytest.approx(0.0, abs=1e-6), f"Expected d=0 at zone center, got {d}"

    def test_tip_inside_zone_slab(self):
        """Tip inside the face zone slab (between face and margin boundary)."""
        cube_pos = torch.zeros(3)
        cube_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        # x = 0.020 is between face (0.015) and outer edge (0.030)
        # zone center x = 0.0225, half_t = 0.0075
        # |0.020 - 0.0225| = 0.0025 < 0.0075, so inside
        tip = torch.tensor([0.020, 0.0, 0.0])
        d, _ = _box_distance_single(tip, cube_pos, cube_quat, use_x=True, sign=1.0,
                                     cube_half_size=self.HALF, zone_margin=self.MARGIN)
        assert d == pytest.approx(0.0, abs=1e-6), f"Expected d=0 inside zone, got {d}"

    def test_tip_at_cube_center(self):
        """Tip at cube center should be cube_half_size distance from zone."""
        cube_pos = torch.zeros(3)
        cube_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        tip = torch.zeros(3)
        d, _ = _box_distance_single(tip, cube_pos, cube_quat, use_x=True, sign=1.0,
                                     cube_half_size=self.HALF, zone_margin=self.MARGIN)
        # |0 - 0.0225| - 0.0075 = 0.015, other dims are 0
        assert d == pytest.approx(self.HALF, abs=1e-6), f"Expected d={self.HALF}, got {d}"

    def test_tip_far_away(self):
        """Tip far from cube -> d should be large and positive."""
        cube_pos = torch.zeros(3)
        cube_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        tip = torch.tensor([0.3, 0.0, 0.0])
        d, _ = _box_distance_single(tip, cube_pos, cube_quat, use_x=True, sign=1.0,
                                     cube_half_size=self.HALF, zone_margin=self.MARGIN)
        assert d > 0.25, f"Expected d >> 0, got {d}"

    def test_negative_face(self):
        """Tip at center of -X face zone -> d = 0."""
        cube_pos = torch.zeros(3)
        cube_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        tip = torch.tensor([-0.0225, 0.0, 0.0])
        d, _ = _box_distance_single(tip, cube_pos, cube_quat, use_x=True, sign=-1.0,
                                     cube_half_size=self.HALF, zone_margin=self.MARGIN)
        assert d == pytest.approx(0.0, abs=1e-6), f"Expected d=0 at -X zone center, got {d}"

    def test_y_axis_face(self):
        """Tip at center of +Y face zone (use_x=False) -> d = 0."""
        cube_pos = torch.zeros(3)
        cube_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        tip = torch.tensor([0.0, 0.0225, 0.0])
        d, _ = _box_distance_single(tip, cube_pos, cube_quat, use_x=False, sign=1.0,
                                     cube_half_size=self.HALF, zone_margin=self.MARGIN)
        assert d == pytest.approx(0.0, abs=1e-6), f"Expected d=0 at +Y zone center, got {d}"

    def test_rotated_cube_90deg_yaw(self):
        """Cube rotated 90° around Z. +X local face now points in +Y world."""
        cube_pos = torch.zeros(3)
        yaw = math.pi / 2
        cube_quat = torch.tensor([math.cos(yaw/2), 0.0, 0.0, math.sin(yaw/2)])

        # In world, the +X face zone center should be at world Y = +0.0225
        tip = torch.tensor([0.0, 0.0225, 0.0])
        d, q = _box_distance_single(tip, cube_pos, cube_quat, use_x=True, sign=1.0,
                                     cube_half_size=self.HALF, zone_margin=self.MARGIN)
        assert d == pytest.approx(0.0, abs=1e-4), f"Expected d=0 at rotated zone center, got {d} (q_local={q})"

    def test_rotated_cube_45deg_yaw(self):
        """Cube rotated 45° around Z. Face zone is at 45° angle."""
        cube_pos = torch.zeros(3)
        yaw = math.pi / 4
        cube_quat = torch.tensor([math.cos(yaw/2), 0.0, 0.0, math.sin(yaw/2)])

        # Compute expected world position of +X face zone center
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        face_offset = self.HALF + 0.5 * self.MARGIN
        tip = torch.tensor([cos_y * face_offset, sin_y * face_offset, 0.0])
        d, q = _box_distance_single(tip, cube_pos, cube_quat, use_x=True, sign=1.0,
                                     cube_half_size=self.HALF, zone_margin=self.MARGIN)
        assert d == pytest.approx(0.0, abs=1e-4), f"Expected d=0 at 45° zone center, got {d} (q_local={q})"

    def test_offset_cube_position(self):
        """Cube at non-origin position, tip at zone center -> d = 0."""
        cube_pos = torch.tensor([0.1, -0.2, 0.015])
        cube_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        # Zone center in world: cube_pos + [0.0225, 0, 0]
        tip = cube_pos + torch.tensor([0.0225, 0.0, 0.0])
        d, _ = _box_distance_single(tip, cube_pos, cube_quat, use_x=True, sign=1.0,
                                     cube_half_size=self.HALF, zone_margin=self.MARGIN)
        assert d == pytest.approx(0.0, abs=1e-6), f"Expected d=0 at offset cube zone, got {d}"

    def test_z_offset_creates_distance(self):
        """Tip above the zone slab should have nonzero d due to Z component."""
        cube_pos = torch.zeros(3)
        cube_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        # Zone center X, but Z = 0.025 (above zone slab top at 0.015)
        tip = torch.tensor([0.0225, 0.0, 0.025])
        d, _ = _box_distance_single(tip, cube_pos, cube_quat, use_x=True, sign=1.0,
                                     cube_half_size=self.HALF, zone_margin=self.MARGIN)
        expected_z_contribution = 0.025 - self.HALF  # 0.01
        assert d == pytest.approx(expected_z_contribution, abs=1e-6), \
            f"Expected d={expected_z_contribution} from Z overshoot, got {d}"

    def test_distance_decreases_approaching(self):
        """Distance should monotonically decrease as tip approaches zone."""
        cube_pos = torch.zeros(3)
        cube_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        distances = []
        for x in [0.3, 0.2, 0.1, 0.05, 0.03, 0.0225]:
            tip = torch.tensor([x, 0.0, 0.0])
            d, _ = _box_distance_single(tip, cube_pos, cube_quat, use_x=True, sign=1.0,
                                         cube_half_size=self.HALF, zone_margin=self.MARGIN)
            distances.append(d)

        for i in range(1, len(distances)):
            assert distances[i] <= distances[i-1] + 1e-6, \
                f"Distance should decrease: d[{i}]={distances[i]:.6f} > d[{i-1}]={distances[i-1]:.6f}"
        assert distances[-1] == pytest.approx(0.0, abs=1e-6), \
            f"Should reach d=0 at zone center, got {distances[-1]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
