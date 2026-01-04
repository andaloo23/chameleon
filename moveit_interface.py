"""
MoveIt2 Socket Client for SO-100 Robot
This module provides a lightweight interface to the MoveIt2 Bridge Server.
It does NOT require rclpy, making it compatible with Isaac Sim's bundled Python.
"""
import socket
import json
import time
from typing import Optional, List, Dict, Any

class Point:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z

class Quaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x, self.y, self.z, self.w = x, y, z, w

class Pose:
    def __init__(self):
        self.position = Point()
        self.orientation = Quaternion()

class MoveItInterface:
    """Lightweight Socket client for MoveIt2 motion planning."""
    
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.sock = None
        self._connect()
        
        # Planning group names (for reference)
        self.arm_group = "arm"
        self.gripper_group = "gripper"
        self.arm_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        self.gripper_joint_names = ["gripper"]

    def _connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print(f"[MoveIt] Connected to Bridge Server at {self.host}:{self.port}")
        except Exception as e:
            print(f"[MoveIt] [ERROR] Could not connect to Bridge Server: {e}")
            self.sock = None

    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not self.sock:
            self._connect()
            if not self.sock:
                return {"status": "error", "message": "Not connected"}
        
        try:
            self.sock.sendall(json.dumps(request).encode('utf-8'))
            data = self.sock.recv(65536)
            if not data:
                self.sock = None
                return {"status": "error", "message": "Connection lost"}
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            print(f"[MoveIt] [ERROR] Socket error: {e}")
            self.sock = None
            return {"status": "error", "message": str(e)}

    def publish_joint_state(self, arm_joints: List[float], gripper_joint: float):
        """Publish current joint states to MoveIt via Bridge."""
        request = {
            "command": "publish_joint_state",
            "arm_joints": [float(j) for j in arm_joints],
            "gripper_joint": float(gripper_joint)
        }
        self._send_request(request)

    def plan_to_pose(self, pose_msg: Any) -> Optional[List[List[float]]]:
        """Plan a trajectory to a target pose.
        
        Args:
            pose_msg: We expect an object with 'position' and 'orientation' 
                     attributes, each with x, y, z (and w) fields.
        """
        # Convert ROS-like pose message to dict for JSON transfer
        pose_dict = {
            "pos": [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z],
            "quat": [pose_msg.orientation.x, pose_msg.orientation.y, 
                     pose_msg.orientation.z, pose_msg.orientation.w]
        }
        
        request = {
            "command": "plan_to_pose",
            "pose": pose_dict
        }
        
        resp = self._send_request(request)
        if resp.get("status") == "success":
            return resp.get("trajectory")
        return None

    def compute_ik(self, target_pos: List[float], target_quat: List[float], 
                   seed_joints: Optional[List[float]] = None) -> Optional[List[float]]:
        """Compute inverse kinematics via Bridge."""
        pose_dict = {
            "pos": [float(x) for x in target_pos],
            "quat": [float(x) for x in target_quat]
        }
        
        request = {
            "command": "compute_ik",
            "pose": pose_dict,
            "seed": seed_joints
        }
        
        resp = self._send_request(request)
        if resp.get("status") == "success":
            return resp.get("solution")
        return None

    def destroy_node(self):
        """Mock method for compatibility with rclpy cleanup code."""
        if self.sock:
            self.sock.close()
            self.sock = None

def init_moveit() -> MoveItInterface:
    """Initialize MoveIt bridge client."""
    return MoveItInterface()

def shutdown_moveit(interface: MoveItInterface):
    """Shutdown MoveIt bridge client."""
    interface.destroy_node()
