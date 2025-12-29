"""
MoveIt2 Python Interface for SO-100 Robot
This module provides motion planning capabilities using MoveIt2.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import RobotState, Constraints
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from sensor_msgs.msg import JointState
import numpy as np
from typing import Optional, List, Tuple


class MoveItInterface(Node):
    """MoveIt2 interface for SO-100 robot motion planning."""
    
    def __init__(self):
        super().__init__('so100_moveit_interface')
        
        # Planning group names
        self.arm_group = "arm"
        self.gripper_group = "gripper"
        
        # Joint names (must match URDF)
        self.arm_joint_names = [
            "shoulder_pan",
            "shoulder_lift", 
            "elbow_flex",
            "wrist_flex",
            "wrist_roll"
        ]
        self.gripper_joint_names = ["gripper"]
        
        # Service clients for IK and motion planning
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        
        # Publisher for joint commands
        self.joint_pub = self.create_publisher(JointState, '/joint_commands', 10)
        
        # Wait for services
        self.get_logger().info('Waiting for MoveIt2 services...')
        self.ik_client.wait_for_service(timeout_sec=10.0)
        self.plan_client.wait_for_service(timeout_sec=10.0)
        self.get_logger().info('MoveIt2 services available!')
        
    def compute_ik(self, target_pose: Pose, 
                   current_joint_state: Optional[List[float]] = None) -> Optional[List[float]]:
        """Compute inverse kinematics for a target pose.
        
        Args:
            target_pose: Target end-effector pose
            current_joint_state: Current joint positions (for seeding IK)
            
        Returns:
            Joint positions if IK solution found, None otherwise
        """
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.arm_group
        
        # Set target pose
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose_stamped.pose = target_pose
        request.ik_request.pose_stamped = pose_stamped
        
        # Set current state as seed if provided
        if current_joint_state is not None:
            robot_state = RobotState()
            robot_state.joint_state.name = self.arm_joint_names
            robot_state.joint_state.position = current_joint_state
            request.ik_request.robot_state = robot_state
            
        # Call IK service
        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        
        if future.result() is not None:
            result = future.result()
            if result.error_code.val == 1:  # SUCCESS
                return list(result.solution.joint_state.position)
        return None
    
    def plan_to_pose(self, target_pose: Pose) -> Optional[List[List[float]]]:
        """Plan a trajectory to a target pose.
        
        Args:
            target_pose: Target end-effector pose
            
        Returns:
            List of waypoints (joint positions) if plan found, None otherwise
        """
        request = GetMotionPlan.Request()
        request.motion_plan_request.group_name = self.arm_group
        request.motion_plan_request.num_planning_attempts = 10
        request.motion_plan_request.allowed_planning_time = 5.0
        
        # Set goal constraints from target pose
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose_stamped.pose = target_pose
        
        # Call planning service
        future = self.plan_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        
        if future.result() is not None:
            result = future.result()
            if result.motion_plan_response.error_code.val == 1:  # SUCCESS
                trajectory = result.motion_plan_response.trajectory.joint_trajectory
                waypoints = []
                for point in trajectory.points:
                    waypoints.append(list(point.positions))
                return waypoints
        return None
    
    def plan_to_joint_positions(self, target_joints: List[float]) -> Optional[List[List[float]]]:
        """Plan a trajectory to target joint positions.
        
        Args:
            target_joints: Target joint positions for the arm
            
        Returns:
            List of waypoints (joint positions) if plan found, None otherwise
        """
        request = GetMotionPlan.Request()
        request.motion_plan_request.group_name = self.arm_group
        request.motion_plan_request.num_planning_attempts = 10
        request.motion_plan_request.allowed_planning_time = 5.0
        
        # Set goal as joint constraints
        constraints = Constraints()
        from moveit_msgs.msg import JointConstraint
        for i, name in enumerate(self.arm_joint_names):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = target_joints[i]
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        request.motion_plan_request.goal_constraints.append(constraints)
        
        # Call planning service
        future = self.plan_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        
        if future.result() is not None:
            result = future.result()
            if result.motion_plan_response.error_code.val == 1:  # SUCCESS
                trajectory = result.motion_plan_response.trajectory.joint_trajectory
                waypoints = []
                for point in trajectory.points:
                    waypoints.append(list(point.positions))
                return waypoints
        return None
    
    def create_pose(self, x: float, y: float, z: float,
                   qx: float = 0.0, qy: float = 0.707, 
                   qz: float = 0.0, qw: float = 0.707) -> Pose:
        """Create a Pose message from position and orientation.
        
        Default orientation is pointing downward (for pick operations).
        """
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose


def init_moveit() -> MoveItInterface:
    """Initialize ROS2 and create MoveIt interface."""
    rclpy.init()
    return MoveItInterface()


def shutdown_moveit(interface: MoveItInterface):
    """Shutdown MoveIt interface and ROS2."""
    interface.destroy_node()
    rclpy.shutdown()
