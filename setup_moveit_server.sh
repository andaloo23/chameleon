#!/bin/bash
# MoveIt2 Jazzy Setup Script for SO-100 Robot
# This script will reconstruct the entire ROS2 package to ensure 
# correct parameter types and namespaces.

# 1. Variables
PKG_NAME="so100_moveit_config"
ROBOT_NAME="so_arm100"
WS_DIR="$HOME/ros2_ws"
SRC_DIR="$WS_DIR/src/$PKG_NAME"

echo "=== Starting SO-100 MoveIt2 Setup ==="

# 2. Preparation
mkdir -p "$SRC_DIR/config"
mkdir -p "$SRC_DIR/launch"
mkdir -p "$SRC_DIR/urdf"
mkdir -p "$SRC_DIR/assets"

# 3. Create OMPL Planning Config (CRITICAL: Explicit list syntax [])
cat <<EOF > "$SRC_DIR/config/$ROBOT_NAME.ompl_planning.yaml"
planning_plugin: "ompl_interface/OMPLPlanner"
request_adapters: [
  "default_planner_request_adapters/AddTimeOptimalParameterization",
  "default_planner_request_adapters/FixWorkspaceBounds",
  "default_planner_request_adapters/FixStartStateBounds",
  "default_planner_request_adapters/FixStartStateCollision",
  "default_planner_request_adapters/FixStartStatePathConstraints"
]
start_state_max_bounds_error: 0.1
arm:
  planner_configs: [RRTConnectkConfigDefault, RRTstarkConfigDefault]
  projection_evaluator: "joints(shoulder_pan,shoulder_lift)"
  longest_valid_segment_fraction: 0.005
gripper:
  planner_configs: [RRTConnectkConfigDefault]
  projection_evaluator: "joints(gripper)"
  longest_valid_segment_fraction: 0.005
planner_configs:
  RRTConnectkConfigDefault:
    type: "geometric::RRTConnect"
    range: 0.0
  RRTstarkConfigDefault:
    type: "geometric::RRTstar"
    range: 0.0
EOF

# 4. Create Kinematics Config
cat <<EOF > "$SRC_DIR/config/$ROBOT_NAME.kinematics.yaml"
arm:
  kinematics_solver: "kdl_kinematics_plugin/KDLKinematicsPlugin"
  kinematics_solver_search_resolution: 0.005
  kinematics_solver_timeout: 0.005
EOF

# 5. Create Joint Limits
cat <<EOF > "$SRC_DIR/config/joint_limits.yaml"
joint_limits:
  shoulder_pan: {has_velocity_limits: true, max_velocity: 1.0}
  shoulder_lift: {has_velocity_limits: true, max_velocity: 1.0}
  elbow_flex: {has_velocity_limits: true, max_velocity: 1.0}
  wrist_flex: {has_velocity_limits: true, max_velocity: 1.0}
  wrist_roll: {has_velocity_limits: true, max_velocity: 1.0}
  gripper: {has_velocity_limits: true, max_velocity: 1.0}
EOF

# 6. Create Package.xml
cat <<EOF > "$SRC_DIR/package.xml"
<?xml version="1.0"?>
<package format="3">
  <name>$PKG_NAME</name>
  <version>1.0.0</version>
  <description>MoveIt config for SO-100</description>
  <maintainer email="user@example.com">User</maintainer>
  <license>MIT</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <exec_depend>moveit_ros_move_group</exec_depend>
  <exec_depend>moveit_configs_utils</exec_depend>
  <exec_depend>moveit_kinematics</exec_depend>
  <exec_depend>moveit_planners_ompl</exec_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher</exec_depend>
  <exec_depend>xacro</exec_depend>
  <export><build_type>ament_cmake</build_type></export>
</package>
EOF

# 7. Create CMakeLists.txt
cat <<EOF > "$SRC_DIR/CMakeLists.txt"
cmake_minimum_required(VERSION 3.8)
project($PKG_NAME)
find_package(ament_cmake REQUIRED)
install(DIRECTORY config launch urdf assets DESTINATION share/\${PROJECT_NAME})
ament_package()
EOF

# 8. Create Launch File (Using official MoveItConfigsBuilder)
cat <<EOF > "$SRC_DIR/launch/move_group.launch.py"
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("$ROBOT_NAME", package_name="$PKG_NAME")
        .robot_description(file_path="urdf/so_arm100.urdf")
        .robot_description_semantic(file_path="config/so_arm100.srdf")
        .robot_description_kinematics(file_path="config/$ROBOT_NAME.kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher", 
             parameters=[moveit_config.robot_description, {"use_sim_time": True}]),
        Node(package="moveit_ros_move_group", executable="move_group", 
             parameters=[moveit_config.to_dict(), {"use_sim_time": True}], output="screen")
    ])
EOF

# Note: URDF, SRDF and assets must still be copied from chameleon
# We assume they are available in ~/chameleon

echo "=== Package Structure Created ==="
echo "Please remember to copy URDF, SRDF and Assets to the package as follows:"
echo "cp ~/chameleon/moveit_config/config/so_arm100.srdf $SRC_DIR/config/"
echo "cp ~/chameleon/moveit_config/urdf/so_arm100.urdf $SRC_DIR/urdf/"
echo "cp -r ~/chameleon/assets/* $SRC_DIR/assets/"
