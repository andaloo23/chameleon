"""
Self-Contained MoveIt2 Launch File for SO-100 Robot
This launch file generates a verified YAML parameter file on the fly to 
ensure exact compatibility with ROS2 Jazzy's parameter requirements.
"""
import os
import yaml
import tempfile
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

    # 2. Construct the exact YAML structure MoveIt2 Jazzy expects
    # We use the 'ros__parameters' key to be absolutely explicit
    params_dict = {
        'move_group': {
            'ros__parameters': {
                'robot_description': robot_description,
                'robot_description_semantic': robot_description_semantic,
                # Kinematics
                'robot_description_kinematics': {
                    'arm': {
                        'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
                        'kinematics_solver_search_resolution': 0.005,
                        'kinematics_solver_timeout': 0.005,
                    }
                },
                # Planning Pipeline
                'planning_pipelines': ['ompl'],
                'default_planning_pipeline': 'ompl',
                'ompl': {
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
                        'planner_configs': ['RRTConnectkConfigDefault'],
                        'projection_evaluator': 'joints(shoulder_pan,shoulder_lift)',
                        'longest_valid_segment_fraction': 0.005,
                    },
                    'gripper': {
                        'planner_configs': ['RRTConnectkConfigDefault'],
                        'projection_evaluator': 'joints(gripper)',
                        'longest_valid_segment_fraction': 0.005,
                    },
                    'planner_configs': {
                        'RRTConnectkConfigDefault': {
                            'type': 'geometric::RRTConnect',
                            'range': 0.0
                        }
                    }
                },
                # Joint Limits
                'robot_description_planning': {
                    'joint_limits': {
                        'shoulder_pan': {'has_velocity_limits': True, 'max_velocity': 1.0},
                        'shoulder_lift': {'has_velocity_limits': True, 'max_velocity': 1.0},
                        'elbow_flex': {'has_velocity_limits': True, 'max_velocity': 1.0},
                        'wrist_flex': {'has_velocity_limits': True, 'max_velocity': 1.0},
                        'wrist_roll': {'has_velocity_limits': True, 'max_velocity': 1.0},
                        'gripper': {'has_velocity_limits': True, 'max_velocity': 1.0},
                    }
                },
                # Performance / Sim settings
                'use_sim_time': True,
                'publish_robot_description': True,
                'publish_robot_description_semantic': True,
                'publish_planning_scene': True,
                'publish_geometry_updates': True,
                'publish_state_updates': True,
                'publish_transforms_updates': True,
                'monitor_dynamics': False,
            }
        }
    }

    # 3. Write to a temporary file on the server
    # This ensures the node reads a clean, fresh, correctly-typed YAML
    tmp_params_file = os.path.join(tempfile.gettempdir(), 'move_group_jazzy_params.yaml')
    with open(tmp_params_file, 'w') as f:
        yaml.dump(params_dict, f)
    
    print(f"Generated temporary parameters at: {tmp_params_file}")

    # 4. Define Nodes
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
        # In ROS2, passing a YAML file path is the most reliable way to preserve structure
        parameters=[tmp_params_file],
    )

    return LaunchDescription([
        robot_state_publisher,
        move_group_node,
    ])
