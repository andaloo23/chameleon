"""
Standard MoveIt2 Launch File for SO-100 Robot
Uses MoveItConfigsBuilder to resolve ROS2 Jazzy parameter namespaces.
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    pkg_name = 'so100_moveit_config'
    
    # MoveItConfigsBuilder automatically looks for:
    # - config/<robot_name>.srdf
    # - config/joint_limits.yaml
    # - config/<robot_name>.kinematics.yaml
    # - config/<robot_name>.ompl_planning.yaml
    moveit_config = (
        MoveItConfigsBuilder("so_arm100", package_name=pkg_name)
        .robot_description(file_path="urdf/so_arm100.urdf")
        .robot_description_semantic(file_path="config/so_arm100.srdf")
        .robot_description_kinematics(file_path="config/so_arm100.kinematics.yaml")
        .joint_limits(file_path="config/so_arm100.joint_limits.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # Move Group Node
    # In Jazzy, name must match what the MoveItConfigsBuilder expects
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
        parameters=[
            moveit_config.robot_description,
            {"use_sim_time": True},
        ],
    )

    return LaunchDescription([
        robot_state_publisher,
        move_group_node,
    ])
