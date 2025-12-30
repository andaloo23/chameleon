"""
Simple MoveIt2 Launch File for SO-100 Robot
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import yaml


def load_yaml(package_name, file_path):
    """Load a yaml file from a package share directory."""
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    with open(absolute_file_path, 'r') as file:
        return yaml.safe_load(file)


def generate_launch_description():
    # Get package directory
    pkg_share = get_package_share_directory('so100_moveit_config')
    
    # Load URDF (use the ROS2-compatible version with package:// paths)
    urdf_file = os.path.join(pkg_share, 'urdf', 'so100_ros.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()
    
    # Load SRDF  
    srdf_file = os.path.join(pkg_share, 'config', 'so100.srdf')
    with open(srdf_file, 'r') as f:
        robot_description_semantic = f.read()
    
    # Load kinematics config
    kinematics_yaml = load_yaml('so100_moveit_config', 'config/kinematics.yaml')
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
        }],
    )
    
    # MoveIt move_group node with proper parameter format for Jazzy
    move_group_params = {
        'robot_description': robot_description,
        'robot_description_semantic': robot_description_semantic,
        'robot_description_kinematics': kinematics_yaml,
        'use_sim_time': True,
        'publish_robot_description': True,
        'publish_robot_description_semantic': True,
        # Planning pipeline - must use correct Jazzy format
        'move_group.planning_pipelines': ['ompl'],
        'move_group.planning_plugin': 'ompl_interface/OMPLPlanner',
        'move_group.request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints',
        'move_group.start_state_max_bounds_error': 0.1,
    }
    
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[move_group_params],
    )
    
    return LaunchDescription([
        robot_state_publisher,
        move_group_node,
    ])
