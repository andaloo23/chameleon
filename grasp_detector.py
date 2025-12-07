"""
Pressure-based grasp detection for robotic grippers.

Detects grasping by identifying "stall" conditions where:
1. The gripper was closing (position was decreasing)
2. The gripper has now stalled (position stable but not at minimum)
3. This indicates the gripper has contacted and is gripping an object
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class GraspState:
    """Current state of grasp detection."""
    grasped: bool = False
    confidence: float = 0.0
    stable_frames: int = 0
    closing_intent: bool = False
    position_blocked: bool = False
    position_stable: bool = False
    gripper_position: float = 0.0
    target_position: float = 0.0
    was_closing: bool = False  # Was the gripper closing before it stalled?


class GraspDetector:
    """
    Detects grasping using position stall detection.
    
    Instead of a fixed gripper threshold (which fails for different object sizes),
    this detector identifies when the gripper:
    1. Was actively closing (position decreasing over time)
    2. Has now stalled (position stable)
    3. Is not at the fully closed position (blocked by object)
    
    This indicates the gripper has contacted and is gripping an object.
    """
    
    def __init__(
        self,
        history_length: int = 15,
        stability_threshold: float = 0.015,
        min_stable_frames: int = 5,
        min_gripper_position: float = 0.08,
        max_gripper_for_grasp: float = 1.0,
        closing_velocity_threshold: float = 0.002,
    ):
        """
        Initialize the grasp detector.
        
        Args:
            history_length: Number of frames to keep in position history
            stability_threshold: Max position change to consider "stable" 
            min_stable_frames: Frames the gripper must be stable to detect grasp
            min_gripper_position: Minimum gripper position (fully closed with no object)
            max_gripper_for_grasp: Maximum gripper position that could indicate a grasp
            closing_velocity_threshold: Minimum velocity to consider "closing"
        """
        self.history_length = history_length
        self.stability_threshold = stability_threshold
        self.min_stable_frames = min_stable_frames
        self.min_gripper_position = min_gripper_position
        self.max_gripper_for_grasp = max_gripper_for_grasp
        self.closing_velocity_threshold = closing_velocity_threshold
        
        # Position history buffer
        self._position_history: deque = deque(maxlen=history_length)
        self._target_history: deque = deque(maxlen=history_length)
        
        # State tracking
        self._stable_frame_count: int = 0
        self._grasped: bool = False
        self._grasp_position: Optional[float] = None
        self._was_closing: bool = False  # Track if we saw closing motion before stall
        self._closing_detected_at_position: Optional[float] = None  # Position when we detected closing
        
    def reset(self) -> None:
        """Reset detector state for a new episode."""
        self._position_history.clear()
        self._target_history.clear()
        self._stable_frame_count = 0
        self._grasped = False
        self._grasp_position = None
        self._was_closing = False
        self._closing_detected_at_position = None
        
    def update(
        self,
        gripper_position: float,
        target_position: Optional[float] = None,
    ) -> GraspState:
        """
        Update detector with new gripper state and return grasp detection result.
        
        Args:
            gripper_position: Current gripper joint position
            target_position: Commanded/target gripper position (if available)
            
        Returns:
            GraspState with current detection status
        """
        # Store in history
        self._position_history.append(gripper_position)
        if target_position is not None:
            self._target_history.append(target_position)
        
        # Need enough history for analysis
        if len(self._position_history) < 5:
            return GraspState(
                grasped=self._grasped,
                gripper_position=gripper_position,
                target_position=target_position or gripper_position,
            )
        
        # Check if gripper was/is actively closing
        is_currently_closing = self._check_is_closing()
        
        # Track if we've seen closing motion
        if is_currently_closing and not self._was_closing:
            self._was_closing = True
            self._closing_detected_at_position = gripper_position
        
        # Check if position has stabilized (stalled)
        position_stable = self._check_position_stability()
        
        # Check if gripper is blocked (not at fully closed position)
        position_blocked = self._check_position_blocked(gripper_position)
        
        # Closing intent: either currently closing, or was closing and now stalled
        closing_intent = is_currently_closing or (self._was_closing and position_stable)
        
        # Grasp detection: was closing, now stable, and not at min position
        grasp_conditions_met = (
            self._was_closing and 
            position_stable and 
            position_blocked
        )
        
        # Update stable frame counter
        if grasp_conditions_met:
            self._stable_frame_count += 1
        else:
            if not self._grasped:
                # Only decay if we haven't grasped yet
                self._stable_frame_count = max(0, self._stable_frame_count - 1)
                # Reset was_closing if gripper opens back up significantly
                if self._closing_detected_at_position is not None:
                    if gripper_position > self._closing_detected_at_position + 0.1:
                        self._was_closing = False
                        self._closing_detected_at_position = None
        
        # Grasp detection with hysteresis
        confidence = min(1.0, self._stable_frame_count / self.min_stable_frames)
        
        if not self._grasped:
            # Detect new grasp
            if self._stable_frame_count >= self.min_stable_frames:
                self._grasped = True
                self._grasp_position = gripper_position
                print(f"[GRASP DETECTOR] ✓ Grasp detected at position {gripper_position:.4f}")
        else:
            # Check for grasp release
            if self._check_grasp_released(gripper_position):
                self._grasped = False
                self._grasp_position = None
                self._stable_frame_count = 0
                self._was_closing = False
                self._closing_detected_at_position = None
        
        return GraspState(
            grasped=self._grasped,
            confidence=confidence,
            stable_frames=self._stable_frame_count,
            closing_intent=closing_intent,
            position_blocked=position_blocked,
            position_stable=position_stable,
            gripper_position=gripper_position,
            target_position=target_position or gripper_position,
            was_closing=self._was_closing,
        )
    
    def _check_is_closing(self) -> bool:
        """Check if gripper is actively closing (position decreasing)."""
        if len(self._position_history) < 5:
            return False
        
        positions = list(self._position_history)
        
        # Check recent velocity (last few frames)
        recent = positions[-5:]
        
        # Calculate average velocity over recent frames
        velocities = [recent[i] - recent[i+1] for i in range(len(recent)-1)]
        avg_velocity = np.mean(velocities)
        
        # Closing = positive velocity (position decreasing, since closing = smaller value)
        return avg_velocity > self.closing_velocity_threshold
    
    def _check_position_stability(self) -> bool:
        """Check if gripper position has been stable (stalled)."""
        if len(self._position_history) < 5:
            return False
        
        recent_positions = list(self._position_history)[-7:]
        if len(recent_positions) < 5:
            return False
        
        # Check range of recent positions
        position_range = max(recent_positions) - min(recent_positions)
        
        # Stable = low range
        return position_range < self.stability_threshold
    
    def _check_position_blocked(self, gripper_position: float) -> bool:
        """Check if gripper is blocked by an object (not at fully closed position)."""
        return (
            gripper_position > self.min_gripper_position
            and gripper_position < self.max_gripper_for_grasp
        )
    
    def _check_grasp_released(self, current_position: float) -> bool:
        """Check if a previously detected grasp has been released."""
        if self._grasp_position is None:
            return True
        
        # Grasp is released if gripper opens significantly from grasp position
        release_threshold = 0.2
        return current_position > self._grasp_position + release_threshold
    
    @property
    def is_grasped(self) -> bool:
        """Return whether an object is currently grasped."""
        return self._grasped
    
    @property
    def grasp_position(self) -> Optional[float]:
        """Return the gripper position at which grasp was detected."""
        return self._grasp_position
