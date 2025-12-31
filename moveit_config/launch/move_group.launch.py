"""
MoveIt2 Launch File for SO-100 Robot
Uses MoveItConfigsBuilder for robust parameter loading in Jazzy.
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    pkg_name = "so100_moveit_config"
    pkg_share = get_package_share_directory(pkg_name)

    moveit_config = (
        MoveItConfigsBuilder("so_arm100", package_name=pkg_name)
        .robot_description(file_path="urdf/so_arm100.urdf")
        .robot_description_semantic(file_path="config/so_arm100.srdf")
        .robot_description_kinematics(file_path="config/so_arm100.kinematics.yaml")
        .joint_limits(file_path="config/so_arm100.joint_limits.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .planning_scene_monitor(
            publish_planning_scene=True,
            publish_geometry_updates=True,
            publish_state_updates=True,
            publish_transforms_updates=True,
        )
        .to_moveit_configs()
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

    return LaunchDescription([
        robot_state_publisher,
        move_group_node,
    ])
