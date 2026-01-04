"""
Monolithic MoveIt2 Launch File for SO-100 Robot
This launch file contains all planning parameters directly in Python to avoid
sync issues with YAML files and ensure correct parameter types for ROS2 Jazzy.
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_name = 'so100_moveit_config'
    pkg_share = get_package_share_directory(pkg_name)
    
    # 1. Load URDF and SRDF
    urdf_file = os.path.join(pkg_share, 'urdf', 'so_arm100.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()
    
    srdf_file = os.path.join(pkg_share, 'config', 'so_arm100.srdf')
    with open(srdf_file, 'r') as f:
        robot_description_semantic = f.read()

    # 2. Define Kinematics (Internalized)
    kinematics_config = {
        'arm': {
            'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
            'kinematics_solver_search_resolution': 0.005,
            'kinematics_solver_timeout': 0.005,
        }
    }

    # 3. Define OMPL Planning (Internalized)
    # This structure is critical for ROS2 Jazzy
    ompl_planning_config = {
        'planning_plugin': 'ompl_interface/OMPLPlanner',
        'request_adapters': [
            'default_planner_request_adapters/AddTimeOptimalParameterization',
            'default_planner_request_adapters/FixWorkspaceBounds',
            'default_planner_request_adapters/FixStartStateBounds',
            'default_planner_request_adapters/FixStartStateCollision',
            'default_planner_request_adapters/FixStartStatePathConstraints',
        ],
        'start_state_max_bounds_error': 0.1,
        'arm': {
            'planner_configs': ['RRTConnectkConfigDefault', 'RRTstarkConfigDefault'],
            'projection_evaluator': 'joints(shoulder_pan,shoulder_lift)',
            'longest_valid_segment_fraction': 0.005,
        },
        'gripper': {
            'planner_configs': ['RRTConnectkConfigDefault'],
            'projection_evaluator': 'joints(gripper)',
            'longest_valid_segment_fraction': 0.005,
        },
        'planner_configs': {
            'RRTConnectkConfigDefault': {'type': 'geometric::RRTConnect', 'range': 0.0},
            'RRTstarkConfigDefault': {'type': 'geometric::RRTstar', 'range': 0.0},
        }
    }

    # 4. Joint Limits (Internalized)
    joint_limits_config = {
        'robot_description_planning': {
            'joint_limits': {
                'shoulder_pan': {'has_velocity_limits': True, 'max_velocity': 1.0, 'has_acceleration_limits': False},
                'shoulder_lift': {'has_velocity_limits': True, 'max_velocity': 1.0, 'has_acceleration_limits': False},
                'elbow_flex': {'has_velocity_limits': True, 'max_velocity': 1.0, 'has_acceleration_limits': False},
                'wrist_flex': {'has_velocity_limits': True, 'max_velocity': 1.0, 'has_acceleration_limits': False},
                'wrist_roll': {'has_velocity_limits': True, 'max_velocity': 1.0, 'has_acceleration_limits': False},
                'gripper': {'has_velocity_limits': True, 'max_velocity': 1.0, 'has_acceleration_limits': False},
            }
        }
    }

    # 5. Combine everything for move_group
    move_group_params = {
        'robot_description': robot_description,
        'robot_description_semantic': robot_description_semantic,
        'robot_description_kinematics': kinematics_config,
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': ompl_planning_config,
        'use_sim_time': True,
        'publish_robot_description': True,
        'publish_robot_description_semantic': True,
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }
    move_group_params.update(joint_limits_config)

    # 6. Nodes
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
    )

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        name='move_group',
        output='screen',
        parameters=[move_group_params],
    )

    return LaunchDescription([
        robot_state_publisher,
        move_group_node,
    ])
