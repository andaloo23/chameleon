"""
ROS2 MoveIt Bridge Server
Runs in system Python (3.12) to provide access to ROS2 Jazzy services
for clients running in different Python versions (like Isaac Sim).
"""
import rclpy
from rclpy.node import Node
import socket
import json
import threading
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import RobotState, Constraints
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from sensor_msgs.msg import JointState
from typing import Optional, List

class MoveItBridgeServer(Node):
    def __init__(self, host='127.0.0.1', port=65432):
        super().__init__('moveit_bridge_server')
        self.host = host
        self.port = port
        
        # Planning groups
        self.arm_group = "arm"
        self.arm_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        self.gripper_joint_names = ["gripper"]
        
        # Service clients
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')
        self.plan_client = self.create_client(GetMotionPlan, 'plan_kinematic_path')
        
        # Publisher for joint states
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Wait for services
        self.get_logger().info('Waiting for MoveIt services...')
        self.ik_client.wait_for_service(timeout_sec=10.0)
        self.plan_client.wait_for_service(timeout_sec=10.0)
        self.get_logger().info('MoveIt services available.')
        
        # Start TCP server
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        self.get_logger().info(f'Bridge Server listening on {host}:{port}')

    def _handle_client(self, conn, addr):
        with conn:
            self.get_logger().info(f'Connected by {addr}')
            while True:
                data = conn.recv(65536)
                if not data:
                    break
                try:
                    request = json.loads(data.decode('utf-8'))
                    cmd = request.get('command')
                    
                    if cmd == 'publish_joint_state':
                        self._publish_joint_state(request['arm_joints'], request['gripper_joint'])
                        response = {'status': 'success'}
                    
                    elif cmd == 'plan_to_pose':
                        traj = self._plan_to_pose(request['pose'])
                        response = {'status': 'success', 'trajectory': traj} if traj else {'status': 'failed'}
                        
                    elif cmd == 'compute_ik':
                        sol = self._compute_ik(request['pose'], request.get('seed'))
                        response = {'status': 'success', 'solution': sol} if sol else {'status': 'failed'}
                    
                    else:
                        response = {'status': 'error', 'message': f'Unknown command: {cmd}'}
                    
                    conn.sendall(json.dumps(response).encode('utf-8'))
                except Exception as e:
                    self.get_logger().error(f'Error handling request: {e}')
                    conn.sendall(json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8'))

    def _run_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            while rclpy.ok():
                conn, addr = s.accept()
                threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()

    def _publish_joint_state(self, arm_joints, gripper_joint):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.arm_joint_names + self.gripper_joint_names
        msg.position = [float(j) for j in arm_joints] + [float(gripper_joint)]
        self.joint_state_pub.publish(msg)

    def _plan_to_pose(self, pose_dict):
        request = GetMotionPlan.Request()
        request.motion_plan_request.group_name = self.arm_group
        request.motion_plan_request.num_planning_attempts = 10
        request.motion_plan_request.allowed_planning_time = 5.0
        
        constraints = Constraints()
        from moveit_msgs.msg import PositionConstraint, OrientationConstraint
        
        # Simplified planning request (pose goal)
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose_stamped.pose.position.x = float(pose_dict['pos'][0])
        pose_stamped.pose.position.y = float(pose_dict['pos'][1])
        pose_stamped.pose.position.z = float(pose_dict['pos'][2])
        pose_stamped.pose.orientation.x = float(pose_dict['quat'][0])
        pose_stamped.pose.orientation.y = float(pose_dict['quat'][1])
        pose_stamped.pose.orientation.z = float(pose_dict['quat'][2])
        pose_stamped.pose.orientation.w = float(pose_dict['quat'][3])
        
        # We wrap it in a goal constraint
        from moveit_msgs.utils import constraints_from_pose
        request.motion_plan_request.goal_constraints.append(
            constraints_from_pose(pose_stamped, 0.01, 0.01)
        )
        
        future = self.plan_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        
        if future.result():
            resp = future.result().motion_plan_response
            if resp.error_code.val == 1:
                return [list(p.positions) for p in resp.trajectory.joint_trajectory.points]
        return None

    def _compute_ik(self, pose_dict, seed_joints=None):
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.arm_group
        request.ik_request.pose_stamped.header.frame_id = "base"
        request.ik_request.pose_stamped.pose.position.x = float(pose_dict['pos'][0])
        request.ik_request.pose_stamped.pose.position.y = float(pose_dict['pos'][1])
        request.ik_request.pose_stamped.pose.position.z = float(pose_dict['pos'][2])
        request.ik_request.pose_stamped.pose.orientation.x = float(pose_dict['quat'][0])
        request.ik_request.pose_stamped.pose.orientation.y = float(pose_dict['quat'][1])
        request.ik_request.pose_stamped.pose.orientation.z = float(pose_dict['quat'][2])
        request.ik_request.pose_stamped.pose.orientation.w = float(pose_dict['quat'][3])
        
        if seed_joints:
            state = RobotState()
            state.joint_state.name = self.arm_joint_names
            state.joint_state.position = [float(j) for j in seed_joints]
            request.ik_request.robot_state = state
            
        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        
        if future.result():
            resp = future.result()
            if resp.error_code.val == 1:
                return list(resp.solution.joint_state.position)
        return None

def main():
    rclpy.init()
    node = MoveItBridgeServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
