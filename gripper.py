"""gripper.py

Refined gripper logic for Isaac Sim. 
This version removes physics welds (sticky gripper) and implements a behavioral 
grasp detection system based on gripper stall and relative distance stability.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque
import numpy as np


class Gripper:
    """Gripper class handling behavioral grasp detection."""

    def __init__(
        self,
        env,
        *,
        dt: float = 1.0 / 120.0,
        stall_threshold_m: float = 0.001,
        contact_limit_m: float = 0.005, # Position below which it's considered "fully closed"
        distance_stability_threshold: float = 0.002,
        history_len: int = 15,
        debug: bool = True,
    ) -> None:
        self.env = env
        self.dt = float(dt)
        self.stall_threshold_m = stall_threshold_m
        self.contact_limit_m = contact_limit_m
        self.distance_stability_threshold = distance_stability_threshold
        self.debug = bool(debug)

        self._gap_history: deque = deque(maxlen=history_len)
        self._distance_history: deque = deque(maxlen=5) # 5 frames for stability as requested
        self._is_grasped: bool = False

    @property
    def is_grasping(self) -> bool:
        return self._is_grasped

    def reset(self) -> None:
        self._gap_history.clear()
        self._distance_history.clear()
        self._is_grasped = False

    def update(
        self,
        *,
        gripper_value: Optional[float],
        target_gripper: Optional[float],
        gripper_world_pos: Optional[np.ndarray],
        jaw_world_pos: Optional[np.ndarray],
        object_world_pos: Optional[np.ndarray],
        arm_moving: bool = False,
    ) -> bool:
        """
        Updates grasp detection.
        Returns: True if grasped, False otherwise.
        """
        if gripper_world_pos is None or jaw_world_pos is None or object_world_pos is None:
            self._is_grasped = False
            return False

        # 1. Calculate current gap (finger distance)
        gap = float(np.linalg.norm(np.asarray(gripper_world_pos) - np.asarray(jaw_world_pos)))
        self._gap_history.append(gap)

        # 2. Check for stall - is the gripper told to move but doesn't?
        stalled = False
        if len(self._gap_history) >= 5:
            gap_range = max(self._gap_history) - min(self._gap_history)
            stalled = gap_range < self.stall_threshold_m
        
        is_closing_cmd = False
        if gripper_value is not None and target_gripper is not None:
            is_closing_cmd = target_gripper < gripper_value - 1e-3

        # Contact: Stalled while trying to close, and not already at the limit
        is_contacting = stalled and is_closing_cmd and gap > self.contact_limit_m

        # 3. Check for distance stability (grasp detection)
        # Gripper center is midpoint between jaws
        gripper_center = 0.5 * (np.asarray(gripper_world_pos) + np.asarray(jaw_world_pos))
        dist_to_obj = float(np.linalg.norm(gripper_center - np.asarray(object_world_pos)))
        self._distance_history.append(dist_to_obj)

        if len(self._distance_history) >= 5:
            dist_std = np.std(self._distance_history)
            # Constancy check: low standard deviation of distance over 5 frames
            is_dist_constant = dist_std < self.distance_stability_threshold
            
            # Grasp requires constant distance AND the arm actually moving (lifting)
            # Also it should be contacting or already grasped
            if is_dist_constant and arm_moving and (is_contacting or self._is_grasped):
                self._is_grasped = True
            else:
                # If arm is moving but distance isn't constant, or cmd is to open
                if (arm_moving and not is_dist_constant) or (gripper_value is not None and gripper_value > 0.6):
                    self._is_grasped = False

        return self._is_grasped
