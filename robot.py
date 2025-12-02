import os
import numpy as np
from omni.isaac.core.robots import Robot
from isaacsim.asset.importer.urdf import _urdf
import omni.kit.commands
from omni.usd import get_context
from omni.isaac.sensor import Camera
from scipy.spatial.transform import Rotation as R
from pxr import UsdGeom
import random


class SO100Robot:
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
        
        self.joint_limits = {
            "shoulder_pan": (-3.14, 3.14),
            "shoulder_lift": (-2.0, 2.0),
            "elbow_flex": (-2.0, 2.0),
            "wrist_flex": (-1.0, 1.0),
            "wrist_roll": (-3.14, 3.14),
            "gripper": (0.0, 0.04)
        }
        
        if import_config is None:
            import_config = _urdf.ImportConfig()
            import_config.convex_decomp = False
            import_config.fix_base = True
            import_config.make_default_prim = True
            import_config.self_collision = False
            import_config.distance_scale = 2.5
            import_config.density = 0.0
        
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
        self.robot = Robot(prim_path=prim_path, name="so_arm100")
        self.world.scene.add(self.robot)
    
    def create_wrist_camera(self):
        """Create a camera attached to the robot's wrist."""
        wrist_camera_path = f"{self.prim_path}/gripper/wrist_camera_sensor"
        
        self.wrist_camera = Camera(
            prim_path=wrist_camera_path,
            name="wrist_camera",
            frequency=30,
            resolution=(128, 128),
        )
        
        self.world.scene.add(self.wrist_camera)
        self.wrist_camera.initialize()
        
        stage = get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(wrist_camera_path)
        
        camera_schema = UsdGeom.Camera(camera_prim)
        camera_schema.GetFocalLengthAttr().Set(18.0)
    
    def update_wrist_camera_position(self, verbose=False):
        """Update the wrist camera position relative to the gripper.
        
        Args:
            verbose: Whether to print debug information
            
        Returns:
            Tuple of (world_position, world_orientation) or (None, None) if failed
        """
        if self.wrist_camera is None:
            return None, None
            
        try:
            local_translation = np.array([0.0, 0.05, -0.08])

            wrist_rot = R.from_matrix(np.array([
                [1, 0, 0],           # X-axis: right
                [0, 0, 1],           # Y-axis: forward  
                [0, -1, 0]           # Z-axis: down
            ]))

            gripper_frame_adjustment = R.from_euler('y', np.pi)

            local_rotation = gripper_frame_adjustment.inv() * wrist_rot * R.from_euler('z', -0.785398163) * R.from_euler('yz', [np.pi, np.pi])
            local_quat = local_rotation.as_quat()
            self.wrist_camera.set_local_pose(
                translation=local_translation,
                orientation=np.array([local_quat[0], local_quat[1], local_quat[2], local_quat[3]])
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
    
    def set_joint_positions(self, positions):
        """Set the joint positions of the robot.
        
        Args:
            positions: Array or list of joint positions
        """
        self.robot.set_joint_positions(np.array(positions))
    
    def get_robot(self):
        """Get the underlying Isaac Sim Robot object.
        
        Returns:
            Robot object
        """
        return self.robot
