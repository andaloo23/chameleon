"""
Simple MoveIt2 Launch File for SO-100 Robot
Uses YAML parameter file for proper MoveIt2 Jazzy configuration.
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import yaml


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
    
    # Load kinematics yaml
    kinematics_file = os.path.join(pkg_share, 'config', 'kinematics.yaml')
    with open(kinematics_file, 'r') as f:
        kinematics_yaml = yaml.safe_load(f)
    
    # Load move_group params yaml
    moveit_params_file = os.path.join(pkg_share, 'config', 'moveit_params.yaml')
    with open(moveit_params_file, 'r') as f:
        moveit_params = yaml.safe_load(f)
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
        }],
    )
    
    # Combine all parameters for move_group
    move_group_params = moveit_params.get('move_group', {})
    move_group_params['robot_description'] = robot_description
    move_group_params['robot_description_semantic'] = robot_description_semantic
    move_group_params['robot_description_kinematics'] = kinematics_yaml
    
    # MoveIt move_group node
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
