"""
MoveIt2 Launch File for SO-100 Robot
Launches MoveIt move_group node with SO-100 configuration.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('so100_moveit_config')
    
    # Build MoveIt config
    moveit_config = (
        MoveItConfigsBuilder("so_arm100", package_name="so100_moveit_config")
        .robot_description(file_path=os.path.join(pkg_dir, "urdf", "so100.urdf"))
        .robot_description_semantic(file_path=os.path.join(pkg_dir, "config", "so100.srdf"))
        .robot_description_kinematics(file_path=os.path.join(pkg_dir, "config", "kinematics.yaml"))
        .joint_limits(file_path=os.path.join(pkg_dir, "config", "joint_limits.yaml"))
        .to_moveit_configs()
    )
    
    # Move Group Node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": True},
        ],
    )
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )
    
    return LaunchDescription([
        robot_state_publisher,
        move_group_node,
    ])
