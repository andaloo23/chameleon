import os
import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from isaacsim.asset.importer.urdf import _urdf
import omni.kit.commands
from omni.usd import get_context
from omni.isaac.sensor import Camera
from scipy.spatial.transform import Rotation as R
from pxr import UsdGeom
import random
from kinematics import KinematicsModel
from typing import Dict, Any, Tuple

class SO100Robot:
    CONFIG = {
        "default": {
            "L1": 117.0,  # Shoulder to elbow length (mm)
            "L2": 223.0,  # Elbow to wrist + gripper length (mm) - 136 + 87
            "BASE_HEIGHT_MM": 120.0,
            "SHOULDER_MOUNT_OFFSET_MM": 32.0,
            "ELBOW_MOUNT_OFFSET_MM": 4.0,
            "SPATIAL_LIMITS": {
                "x": (-20.0, 350.0),
                "z": (10.0, 450.0),
            }
        },
        "PRESET_POSITIONS": {
            "1": { "gripper": 0.0, "wrist_roll": 90.0, "wrist_flex": 0.0, "elbow_flex": 0.0, "shoulder_lift": 0.0, "shoulder_pan": 90.0 },
            "2": { "gripper": 0.0, "wrist_roll": 90.0, "wrist_flex": 0.0, "elbow_flex": 45.0, "shoulder_lift": 45.0, "shoulder_pan": 90.0 },
            "3": { "gripper": 40.0, "wrist_roll": 90.0, "wrist_flex": 90.0, "elbow_flex": 45.0, "shoulder_lift": 45.0, "shoulder_pan": 90.0 },
            "4": { "gripper": 40.0, "wrist_roll": 90.0, "wrist_flex": -60.0, "elbow_flex": 20.0, "shoulder_lift": 80.0, "shoulder_pan": 90.0 },
        },
        "MOVEMENT_CONSTANTS": {
            "DEGREES_PER_STEP": 1.5,           # Degrees per interpolation step
            "MAX_INTERPOLATION_STEPS": 150,    # Maximum number of interpolation steps
            "STEP_DELAY_SECONDS": 0.01,        # Delay between interpolation steps (100Hz)
        }
    }

    def __init__(self, world, urdf_path, import_config=None):
        """Initialize the SO-100 robotic arm.
        
        Args:
            world: Isaac Sim World object
            urdf_path: Path to the URDF file
            import_config: Optional URDF import configuration
        """
        self.world = world
        self.urdf_path = urdf_path
        self.wrist_camera = None
        
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        
        # Initialize kinematics model
        self.kinematics = KinematicsModel(self.CONFIG["default"])
        
        # Joint limits (radians) - derived from Hardware degrees (0 to 180, centered at 90)
        # Simulation uses radians where 0 is often the "natural" URDF position.
        # We'll map hardware 90 degrees to simulation 0 radians.
        self.joint_limits = {
            "shoulder_pan": (-np.deg2rad(90), np.deg2rad(90)),
            "shoulder_lift": (0.0, 3.5),
            "elbow_flex": (-3.14159, 0.0),
            "wrist_flex": (-2.5, 1.2),
            "wrist_roll": (-3.14159, 3.14159),
            "gripper": (0.0, 1.5)
        }
        
        if import_config is None:
            import_config = _urdf.ImportConfig()
            import_config.convex_decomp = True   # Enable convex decomposition for proper collision
            import_config.fix_base = True
            import_config.make_default_prim = True
            import_config.self_collision = False  # Keep self-collision off to avoid arm conflicts
            import_config.distance_scale = 1.0
            import_config.density = 500.0  # Set some density for collision response
        
        self.import_config = import_config
        
        self._load_robot()
    
    def _load_robot(self):
        """Load the robot from URDF file."""
        urdf_interface = _urdf.acquire_urdf_interface()
        
        result, robot_model = omni.kit.commands.execute(
            "URDFParseFile",
            urdf_path=self.urdf_path,
            import_config=self.import_config
        )
        
        result, prim_path = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_robot=robot_model,
            import_config=self.import_config,
        )
        
        self.prim_path = prim_path
        self.robot = Articulation(prim_path=prim_path, name="so_arm100")
        self.world.scene.add(self.robot)
        
    def configure_drives(self):
        """Configure joint drives for smooth motion using USD API."""
        from pxr import UsdPhysics, PhysxSchema
        stage = self.world.stage
        
        # Define joints and their desired gains
        # joint_name -> (stiffness, damping)
        arm_gains = {
            "shoulder_pan": (1e6, 1e4),
            "shoulder_lift": (1e6, 1e4),
            "elbow_flex": (1e6, 1e4),
            "wrist_flex": (1e6, 1e4),
            "wrist_roll": (1e6, 1e4),
            "gripper": (6000.0, 400.0)
        }
        
        for joint_name, (stiffness, damping) in arm_gains.items():
            joint_path = f"{self.prim_path}/{joint_name}"
            joint_prim = stage.GetPrimAtPath(joint_path)
            if joint_prim.IsValid():
                if not joint_prim.HasAPI(PhysxSchema.PhysxJointAPI):
                    PhysxSchema.PhysxJointAPI.Apply(joint_prim)
                
                # Apply DriveAPI for angular drive
                drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
                drive.CreateStiffnessAttr().Set(float(stiffness))
                drive.CreateDampingAttr().Set(float(damping))
                drive.CreateMaxForceAttr().Set(100.0)
            else:
                # Try common variations if name doesn't match exactly
                for child in stage.GetPrimAtPath(self.prim_path).GetChildren():
                    if joint_name in child.GetName():
                        if not child.HasAPI(PhysxSchema.PhysxJointAPI):
                            PhysxSchema.PhysxJointAPI.Apply(child)
                        drive = UsdPhysics.DriveAPI.Apply(child, "angular")
                        drive.CreateStiffnessAttr().Set(float(stiffness))
                        drive.CreateDampingAttr().Set(float(damping))
                        drive.CreateMaxForceAttr().Set(100.0)
                        break
    
    def create_wrist_camera(self):
        """Create a camera attached to the robot's wrist."""
        wrist_camera_path = f"{self.prim_path}/gripper/wrist_camera_sensor"
        
        # Verify that the parent link exists
        stage = get_context().get_stage()
        parent_prim = stage.GetPrimAtPath(f"{self.prim_path}/gripper")
        if not parent_prim.IsValid():
            print(f"[WARN] Wrist camera parent link '/gripper' not found at {self.prim_path}. Checking children...")
            for child in stage.GetPrimAtPath(self.prim_path).GetChildren():
                if "gripper" in child.GetName().lower():
                    wrist_camera_path = f"{child.GetPath()}/wrist_camera_sensor"
                    print(f"[INFO] Using child link for camera: {child.GetPath()}")
                    break

        self.wrist_camera = Camera(
            prim_path=wrist_camera_path,
            name="wrist_camera",
            frequency=30,
            resolution=(128, 128),
        )
        
        self.world.scene.add(self.wrist_camera)
        self.wrist_camera.initialize()
        
        # Force some updates for camera stability
        for _ in range(5):
            self.world.step(render=True)
            if hasattr(self.world, "app"):
                self.world.app.update()
        
        camera_prim = stage.GetPrimAtPath(wrist_camera_path)
        if camera_prim.IsValid():
            camera_schema = UsdGeom.Camera(camera_prim)
            camera_schema.GetFocalLengthAttr().Set(18.0)
            camera_schema.GetHorizontalApertureAttr().Set(2.0955)
            camera_schema.GetVerticalApertureAttr().Set(2.0955)
    
    def update_wrist_camera_position(self, verbose=False, cycle_rotations=False):
        """Update the wrist camera position relative to the gripper.
        
        Args:
            verbose: Whether to print debug information
            cycle_rotations: If True, cycles through all rotation configurations
            
        Returns:
            Tuple of (world_position, world_orientation) or (None, None) if failed
        """
        if self.wrist_camera is None:
            return None, None
        
        # Initialize rotation cycling state if needed
        if not hasattr(self, '_rotation_cycle_state'):
            self._rotation_cycle_state = {
                'index': 0,
                'last_change_time': None,
                'angles': [0, 90, 180, -90]
            }
            
        try:
            local_translation = np.array([0, 0.1, -5])

            if cycle_rotations:
                import time
                angles = self._rotation_cycle_state['angles']
                current_time = time.time()
                
                # Change configuration every 3 seconds
                if (self._rotation_cycle_state['last_change_time'] is None or 
                    current_time - self._rotation_cycle_state['last_change_time'] >= 3.0):
                    
                    self._rotation_cycle_state['last_change_time'] = current_time
                    idx = self._rotation_cycle_state['index']
                    
                    # Calculate x, y, z indices from flat index (4^3 = 64 combinations)
                    x_idx = idx % 4
                    y_idx = (idx // 4) % 4
                    z_idx = (idx // 16) % 4
                    
                    x_angle = angles[x_idx]
                    y_angle = angles[y_idx]
                    z_angle = angles[z_idx]
                    
                    print(f"\n[CAMERA ROTATION] Config {idx+1}/64: X={x_angle}°, Y={y_angle}°, Z={z_angle}°")
                    
                    # Advance to next configuration
                    self._rotation_cycle_state['index'] = (idx + 1) % 64
                    self._rotation_cycle_state['current_angles'] = (x_angle, y_angle, z_angle)
                
                # Use the current angles from state
                if 'current_angles' in self._rotation_cycle_state:
                    x_angle, y_angle, z_angle = self._rotation_cycle_state['current_angles']
                else:
                    x_angle, y_angle, z_angle = 0, 0, 0
                    
                local_rotation = R.from_euler('xyz', [x_angle, y_angle, z_angle], degrees=True)
            else:
                # Default rotation: X=90°, Y=-90°, Z=0°
                local_rotation = R.from_euler('xyz', [90, -90, 0], degrees=True)
            
            local_quat = local_rotation.as_quat() # returns [x, y, z, w]
            
            # Camera orientation in Isaac Sim is [w, x, y, z]
            self.wrist_camera.set_local_pose(
                translation=local_translation,
                orientation=np.array([local_quat[3], local_quat[0], local_quat[1], local_quat[2]])
            )
            
            if verbose:
                print(f"Camera quaternion: {local_quat}")

                world_pos, world_orient = self.wrist_camera.get_world_pose()
                print(f"Camera local transform set. World pose: pos={world_pos}, orient={world_orient}")
                
                fov = self.wrist_camera.get_horizontal_fov()
                print(f"Camera FOV: {fov} degrees")
            
            return self.wrist_camera.get_world_pose()
                    
        except Exception as e:
            if verbose:
                print(f"Error setting camera local pose: {e}")
        
        return None, None
    
    def get_random_joint_positions(self):
        """Generate small random joint deltas while respecting joint limits."""
        try:
            current_positions = np.array(self.robot.get_joint_positions(), dtype=float)
        except Exception:
            current_positions = None

        if current_positions is None or current_positions.size != len(self.joint_names):
            current_positions = np.array([
                0.5 * (self.joint_limits[name][0] + self.joint_limits[name][1])
                for name in self.joint_names
            ], dtype=float)

        new_positions = []
        for idx, name in enumerate(self.joint_names):
            lower, upper = self.joint_limits[name]
            span = max(upper - lower, 1e-6)
            max_delta = min(0.25, 0.1 * span)
            delta = random.uniform(-max_delta, max_delta)
            updated = float(np.clip(current_positions[idx] + delta, lower, upper))
            new_positions.append(updated)

        return new_positions
    
    def set_joint_positions(self, positions, use_targets=True):
        """Set the joint positions of the robot.
        
        Args:
            positions: Array or list of joint positions
            use_targets: If True, uses PD control targets via apply_action. If False, snaps immediately.
        """
        if use_targets:
            self.robot.apply_action(ArticulationAction(joint_positions=np.array(positions)))
        else:
            self.robot.set_joint_positions(np.array(positions))
    
    def move_to_preset(self, preset_id: str, use_targets=True):
        """Move the robot to a predefined preset position.
        
        Args:
            preset_id: Key from PRESET_POSITIONS
            use_targets: Whether to use PD control
        """
        if preset_id not in self.CONFIG["PRESET_POSITIONS"]:
            print(f"[ERROR] Preset {preset_id} not found.")
            return

        preset = self.CONFIG["PRESET_POSITIONS"][preset_id]
        joint_positions = []
        
        for name in self.joint_names:
            deg_val = preset.get(name, 90.0) # Hardware uses 90 as center
            if name == "gripper":
                # Gripper is special, typically 0.0 (closed) to some max value (open)
                # In simulation it's 0 to 1.5. Hardware might be 0 to 100.
                # Assuming hardware 0 is closed, 100 is open.
                rad_val = (deg_val / 40.0) * 1.5 # 40 is max in presets
            else:
                # Map 90 deg hardware to 0 rad simulation
                rad_val = np.deg2rad(deg_val - 90.0)
            
            joint_positions.append(rad_val)
        
        self.set_joint_positions(joint_positions, use_targets=use_targets)

    def move_to_cartesian(self, x: float, z: float, shoulder_pan_deg: float = 90.0, wrist_roll_deg: float = 90.0, gripper_val: float = 0.0):
        """Move the end-effector to a Cartesian X, Z position using IK.
        
        Args:
            x: Forward/backward distance in mm
            z: Up/down distance in mm
            shoulder_pan_deg: Pan angle in degrees (Hardware 90 = Center)
            wrist_roll_deg: Roll angle in degrees (Hardware 90 = Center)
            gripper_val: Gripper value
        """
    def get_ik_joints(self, x, z, shoulder_pan_deg=90.0, wrist_roll_deg=90.0, wrist_flex_deg=90.0, gripper_val=0.0):
        """Calculate joint positions for a Cartesian target (X, Z) in mm.
        
        Args:
            x: Horizontal distance from base (mm)
            z: Vertical height from base (mm)
            shoulder_pan_deg: Rotation of base (Hardware 90 = Center)
            wrist_roll_deg: Roll angle (Hardware 90 = Center)
            wrist_flex_deg: Pitch angle (Hardware 90 = Center)
            gripper_val: Hardware gripper value (0-40)
        """
        valid, msg = self.kinematics.is_cartesian_target_valid(x, z)
        if not valid:
            print(f"[WARN] IK Target invalid: {msg}")
            return None
        
        shoulder_lift_deg, elbow_flex_deg = self.kinematics.inverse_kinematics(x, z)
        
        joint_dict = {
            "shoulder_pan": np.deg2rad(shoulder_pan_deg - 90.0),
            "shoulder_lift": np.deg2rad(shoulder_lift_deg),
            "elbow_flex": np.deg2rad(elbow_flex_deg),
            "wrist_flex": np.deg2rad(wrist_flex_deg - 90.0) + 1.0, # Offset to compensate for URDF rpy="-1 0 0"
            "wrist_roll": np.deg2rad(wrist_roll_deg - 90.0),
            "gripper": (gripper_val / 40.0) * 1.5
        }
        
        return [joint_dict[name] for name in self.joint_names]

    def move_to_cartesian(self, x, z, shoulder_pan_deg=90.0, wrist_roll_deg=90.0, gripper_val=0.0):
        positions = self.get_ik_joints(x, z, shoulder_pan_deg, wrist_roll_deg, 90.0, gripper_val)
        if positions:
            self.set_joint_positions(positions)
            return True
        return False

    def calculate_ik_from_world(self, world_pos):
        """Map world (X, Y, Z) in meters to robot-relative (x_mm, z_mm, pan_deg)."""
        wx, wy, wz = world_pos
        # Robot is at origin (0, 0, 0)
        # Horizontal dist in meters
        dist_m = np.sqrt(wx**2 + wy**2)
        # Height from base in meters
        # URDF Base Height is ~120mm = 0.120m
        height_m = wz 
        
        # Pan angle: atan2(x, -y). Hardware 90 is center (facing -Y)
        pan_rad = np.arctan2(wx, -wy) 
        pan_deg = 90.0 + np.rad2deg(pan_rad)
        
        # Clamp pan to reachable range (0-180 for hardware)
        pan_deg = np.clip(pan_deg, 0.0, 180.0)
        
        return dist_m * 1000.0, height_m * 1000.0, pan_deg

    def move_interpolated(self, target_positions, render=True):
        """Move to target positions using interpolation constants.
        
        Args:
            target_positions: Final joint positions (radians)
            render: Whether to render simulation steps
        """
        current_positions = np.array(self.robot.get_joint_positions())
        target_positions = np.array(target_positions)
        
        diff = target_positions - current_positions
        max_diff_deg = np.max(np.abs(np.rad2deg(diff)))
        
        constants = self.CONFIG["MOVEMENT_CONSTANTS"]
        num_steps = int(min(
            constants["MAX_INTERPOLATION_STEPS"],
            max(1, max_diff_deg / constants["DEGREES_PER_STEP"])
        ))
        
        for i in range(1, num_steps + 1):
            interp_pos = current_positions + (diff * (i / num_steps))
            self.set_joint_positions(interp_pos, use_targets=True)
            self.world.step(render=render)
            
            # Optional: Simulate hardware delay
            # time.sleep(constants["STEP_DELAY_SECONDS"]) 
            # Note: time.sleep might block Isaac Sim execution, 
            # so we rely on world.step() for simulation time.
        
        # Ensure we reached the target exactly at the end
        self.set_joint_positions(target_positions, use_targets=True)
        self.world.step(render=render)
    
    def get_robot(self):
        """Get the underlying Isaac Sim Robot object.
        
        Returns:
            Robot object
        """
        return self.robot
