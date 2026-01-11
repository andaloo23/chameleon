import math
from typing import Tuple, Dict, Any

class KinematicsModel:
    """
    Encapsulates the robot's kinematic model and calculations.
    This includes link lengths, mounting offsets, and spatial limits,
    as well as forward and inverse kinematics functions.
    """
    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the kinematics model with robot-specific parameters.
        """
        self.L1: float = params["L1"]
        self.L2: float = params["L2"]
        self.BASE_HEIGHT_MM: float = params["BASE_HEIGHT_MM"]
        self.SHOULDER_MOUNT_OFFSET_MM: float = params["SHOULDER_MOUNT_OFFSET_MM"]
        self.ELBOW_MOUNT_OFFSET_MM: float = params["ELBOW_MOUNT_OFFSET_MM"]
        self.SPATIAL_LIMITS: Dict[str, Tuple[float, float]] = params["SPATIAL_LIMITS"]
        
        # Calculated offsets in radians
        self.SHOULDER_OFFSET_ANGLE_RAD = math.asin(self.SHOULDER_MOUNT_OFFSET_MM / self.L1)
        self.ELBOW_OFFSET_ANGLE_RAD = math.asin(self.ELBOW_MOUNT_OFFSET_MM / self.L2)

    def forward_kinematics(self, shoulder_lift_deg: float, elbow_flex_deg: float) -> tuple[float, float]:
        """Calculates x, z position of the wrist flex motor based on shoulder_lift and elbow_flex angles."""
        ang1_fk = math.radians(shoulder_lift_deg) + self.SHOULDER_OFFSET_ANGLE_RAD
        ang2_fk = math.radians(elbow_flex_deg) + self.ELBOW_OFFSET_ANGLE_RAD - math.radians(shoulder_lift_deg)
        x = -self.L1 * math.cos(ang1_fk) + self.L2 * math.cos(ang2_fk)
        z =  self.L1 * math.sin(ang1_fk) + self.L2 * math.sin(ang2_fk) + self.BASE_HEIGHT_MM
        return x, z

    def inverse_kinematics(self, target_x: float, target_z: float) -> tuple[float, float]:
        """Calculates shoulder_lift and elbow_flex angles (degrees) for a target X, Z."""
        z_adj = target_z - self.BASE_HEIGHT_MM
        d_sq = target_x**2 + z_adj**2
        d = math.sqrt(d_sq)
        
        # Law of Cosines for the interior angle at the shoulder
        # We want the "elbow up" solution for a "concave down" look.
        # phi1: angle to the target point
        phi1 = math.atan2(z_adj, target_x)
        # phi2: angle between L1 and the line to target
        phi2 = math.acos(min(1.0, max(-1.0, (self.L1**2 + d_sq - self.L2**2) / (2 * self.L1 * d))))
        
        # Consistent "elbow up" solution:
        # shoulder_lift = 180 - (phi1 + phi2) in some frames,
        # but for SO-100 hardware, we want the arch.
        shoulder_lift_deg = 180.0 - math.degrees(phi1 + phi2) - math.degrees(self.SHOULDER_OFFSET_ANGLE_RAD)
        
        alpha1 = math.radians(shoulder_lift_deg) + self.SHOULDER_OFFSET_ANGLE_RAD
        cos2_arg = min(1.0, max(-1.0, (target_x + self.L1 * math.cos(alpha1)) / self.L2))
        sin2_arg = min(1.0, max(-1.0, (z_adj - self.L1 * math.sin(alpha1)) / self.L2))
        ang2 = math.atan2(sin2_arg, cos2_arg)
        elbow_flex_deg = math.degrees(ang2 + math.radians(shoulder_lift_deg)) - math.degrees(self.ELBOW_OFFSET_ANGLE_RAD)
        
        return shoulder_lift_deg, elbow_flex_deg

    def is_cartesian_target_valid(self, x: float, z: float) -> tuple[bool, str]:
        """
        Checks if the x,z position is valid against spatial and reach limits.
        """
        if not (self.SPATIAL_LIMITS["x"][0] <= x <= self.SPATIAL_LIMITS["x"][1]):
            return False, f"Target X {x:.1f}mm out of range {self.SPATIAL_LIMITS['x']}"
        if not (self.SPATIAL_LIMITS["z"][0] <= z <= self.SPATIAL_LIMITS["z"][1]):
            return False, f"Target Z {z:.1f}mm out of range {self.SPATIAL_LIMITS['z']}"
        if x < 20 and z < 150:
            return False, f"Target ({x:.1f},{z:.1f})mm violates: if x < 20mm, z must be >= 150mm."
        
        z_adj = z - self.BASE_HEIGHT_MM
        distance = math.sqrt(z_adj**2 + x**2)
        max_reach = self.L1 + self.L2
        
        if distance > max_reach - 1:
            return False, f"Target ({x:.1f},{z:.1f})mm is beyond max reach {max_reach-1:.1f}mm (safety margin: 1mm), distance is {distance:.1f}mm"
        return True, "Valid" 