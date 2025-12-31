"""
Simplified MoveIt2 Launch File for SO-100 Robot
Manually loads parameters to ensure correct structure for ROS2 Jazzy.
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import yaml


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


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
    
    # 2. Load YAML configs
    kinematics_yaml = load_yaml(pkg_name, 'config/kinematics.yaml')
    joint_limits_yaml = load_yaml(pkg_name, 'config/joint_limits.yaml')
    ompl_yaml = load_yaml(pkg_name, 'config/ompl_planning.yaml')
    
    # 3. Build move_group parameters
    move_group_params = {
        'robot_description': robot_description,
        'robot_description_semantic': robot_description_semantic,
        'robot_description_kinematics': kinematics_yaml,
        'robot_description_planning': joint_limits_yaml,
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': ompl_yaml,
        'use_sim_time': True,
        # Required for MoveGroup to publish its state
        'publish_robot_description': True,
        'publish_robot_description_semantic': True,
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }
    
    # 4. Robot State Publisher
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
    
    # 5. MoveIt move_group node
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
