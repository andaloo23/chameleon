"""
MoveIt2 Launch File for SO-100 Robot
Loads parameters from multiple YAML files for MoveIt2 Jazzy.
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import yaml


def load_yaml(package_name, file_path):
    """Load a yaml file from a package share directory."""
    try:
        package_path = get_package_share_directory(package_name)
        absolute_file_path = os.path.join(package_path, file_path)
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def generate_launch_description():
    pkg_name = 'so100_moveit_config'
    pkg_share = get_package_share_directory(pkg_name)
    
    # 1. Load URDF
    urdf_file = os.path.join(pkg_share, 'urdf', 'so100_ros.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()
    
    # 2. Load SRDF  
    srdf_file = os.path.join(pkg_share, 'config', 'so100.srdf')
    with open(srdf_file, 'r') as f:
        robot_description_semantic = f.read()
    
    # 3. Load Kinematics
    kinematics_yaml = load_yaml(pkg_name, 'config/kinematics.yaml')
    
    # 4. Load MoveGroup parameters
    moveit_params = load_yaml(pkg_name, 'config/moveit_params.yaml')
    move_group_params = moveit_params.get('move_group', {})
    
    # 5. Build full parameter set
    # The node expects these as top-level parameters
    combined_params = {}
    combined_params.update(move_group_params)
    combined_params['robot_description'] = robot_description
    combined_params['robot_description_semantic'] = robot_description_semantic
    combined_params['robot_description_kinematics'] = kinematics_yaml
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
    )
    
    # MoveIt move_group node
    # We name the node 'move_group' to match the parameter file namespace
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        name='move_group',
        output='screen',
        parameters=[combined_params],
    )
    
    return LaunchDescription([
        robot_state_publisher,
        move_group_node,
    ])
