import numpy as np

# Reward weights for 5-stage task
# Stage 1: Approach cube (dense shaping)
APPROACH_DISTANCE_MAX = 1.0   # Max distance for shaping
APPROACH_WEIGHT = 1.0         # Reward weight for approaching cube

# Stage 2: Grasp cube (one-time bonus)
GRASP_BONUS = 2.0

# Stage 3: Transport to cup (dense shaping) 
TRANSPORT_DISTANCE_MAX = 0.5  # Max XY distance for shaping
TRANSPORT_WEIGHT = 2.0        # Higher weight for transport (post-grasp)

# Stage 4: Droppable range (one-time bonus)
DROPPABLE_BONUS = 1.5

# Stage 5: Success (one-time bonus)
SUCCESS_BONUS = 10.0

# Penalties
ACTION_COST_WEIGHT = 0.01     # Per-step action cost
DROP_PENALTY = -3.0           # Dropping cube not in cup
CUP_COLLISION_PENALTY = -0.5  # Per-step penalty for cup collision
JOINT_LIMIT_PENALTY_WEIGHT = 1.0

# Self-collision prevention
SELF_COLLISION_THRESHOLD = 0.05  # 5cm - distance threshold for penalty
SELF_COLLISION_PENALTY_WEIGHT = 2.0  # Weight for self-collision penalty


class RewardEngine:
    def __init__(self, env):
        self.env = env
        self.gripper_joint_index = None

        self.stage_flags = {}
        self.reward_components = {}
        self.drop_detected = False
        self.task_state = {}
        self.latest_joint_positions = None
        self.latest_joint_velocities = None
        self.latest_target_gripper = None  # Target gripper position from action
        
        # Consolidate with env.gripper_detector (Gripper class)
        # to ensure reward logic matches simulation status reporting

    def initialize(self):
        robot = getattr(self.env, "robot", None)
        joint_names = getattr(robot, "joint_names", None)
        if joint_names:
            try:
                self.gripper_joint_index = joint_names.index("gripper")
            except ValueError:
                self.gripper_joint_index = None

    def reset(self):
        self.stage_flags = {
            # Original stage flags
            "grasped": False,           # Stage 2: one-time grasp bonus
            "droppable_reached": False, # Stage 4: one-time droppable range bonus
            "success": False,           # Stage 5: one-time success bonus
            # New milestone flags for PPO tracking
            "reached": False,           # Gripper within reach distance of cube
            "controlled": False,        # Alias for grasped (grasp detected)
            "lifted": False,            # Cube lifted above ground threshold
            "above_cup": False,         # Cube positioned above cup XY
            "released": False,          # Gripper opened after grasp
        }
        self.reward_components = {}
        self.drop_detected = False
        self.task_state = {}
        self.latest_joint_positions = None
        self.latest_joint_velocities = None
        self.latest_target_gripper = None
        self._was_grasping = False  # Track grasp state for release detection
        # env.gripper_detector.reset() is called by the env itself

    def record_joint_state(self, joint_positions, joint_velocities, target_gripper=None):
        """
        Record current joint state and update task state.
        
        Args:
            joint_positions: Current joint positions
            joint_velocities: Current joint velocities
            target_gripper: Optional target gripper position from action command
        """
        self.latest_joint_positions = joint_positions
        self.latest_joint_velocities = joint_velocities
        self.latest_target_gripper = target_gripper
        self._update_task_state()

    def compute_reward_components(self):
        """
        Compute reward components for the 5-stage pick-and-place task:
        
        Stage 1: Approach cube - dense shaping reward for getting closer to cube
        Stage 2: Grasp cube - one-time bonus when grasp detected
        Stage 3: Transport to cup - dense shaping for decreasing XY distance to cup
        Stage 4: Droppable range - one-time bonus when cube above cup
        Stage 5: Success - one-time bonus for cube dropped in cup
        
        Penalties:
        - Action cost: per-step cost based on joint velocities
        - Drop penalty: one-time penalty for dropping cube not in cup
        - Cup collision: per-step penalty for gripper hitting cup
        - Joint limit penalty: per-step soft penalty for approaching joint limits
        """
        state = self.task_state
        components = {}
        
        # Get state values
        gripper_cube_distance = state.get("gripper_cube_distance")
        cube_cup_distance_xy = state.get("cube_cup_distance_xy")
        cube_height = state.get("cube_height")
        joint_positions = state.get("joint_positions")
        joint_velocities = state.get("joint_velocities")
        
        # Get detector state
        grasp_detected = getattr(self.env.gripper_detector, "is_grasping", False)
        droppable_detected = getattr(self.env.gripper_detector, "is_droppable_range", False)
        in_cup_detected = getattr(self.env.gripper_detector, "is_in_cup", False)
        cup_collision = getattr(self.env.gripper_detector, "is_cup_collision", False)
        
        # ===== STAGE 1: Approach Cube =====
        # Dense shaping: reward increases as gripper gets closer to cube
        if gripper_cube_distance is not None:
            approach_ratio = 1.0 - min(gripper_cube_distance, APPROACH_DISTANCE_MAX) / APPROACH_DISTANCE_MAX
            approach_ratio = max(0.0, approach_ratio)
            components["approach_shaping"] = APPROACH_WEIGHT * approach_ratio
        else:
            components["approach_shaping"] = 0.0
        
        # ===== STAGE 2: Grasp Cube =====
        # One-time bonus when grasp is detected
        if not self.stage_flags.get("grasped") and grasp_detected:
            self.stage_flags["grasped"] = True
            components["grasp_bonus"] = GRASP_BONUS
        else:
            components["grasp_bonus"] = 0.0
        
        # ===== STAGE 3: Transport to Cup =====
        # Dense shaping: reward increases as cube XY distance to cup decreases
        # Only active after grasping (to avoid rewarding moving cube by pushing)
        if self.stage_flags.get("grasped") and cube_cup_distance_xy is not None:
            transport_ratio = 1.0 - min(cube_cup_distance_xy, TRANSPORT_DISTANCE_MAX) / TRANSPORT_DISTANCE_MAX
            transport_ratio = max(0.0, transport_ratio)
            components["transport_shaping"] = TRANSPORT_WEIGHT * transport_ratio
        else:
            components["transport_shaping"] = 0.0
        
        # ===== STAGE 4: Droppable Range =====
        # One-time bonus when cube is positioned above cup (ready to drop)
        if not self.stage_flags.get("droppable_reached") and droppable_detected:
            self.stage_flags["droppable_reached"] = True
            components["droppable_bonus"] = DROPPABLE_BONUS
        else:
            components["droppable_bonus"] = 0.0
        
        # ===== STAGE 5: Success =====
        # One-time bonus when cube is inside the cup
        if not self.stage_flags.get("success") and in_cup_detected:
            self.stage_flags["success"] = True
            components["success_bonus"] = SUCCESS_BONUS
        else:
            components["success_bonus"] = 0.0
        
        # ===== PENALTIES =====
        
        # Action cost: penalize large joint velocities to encourage efficiency
        if joint_velocities is not None:
            action_cost = float(np.linalg.norm(joint_velocities))
            components["action_cost"] = -ACTION_COST_WEIGHT * action_cost
        else:
            components["action_cost"] = 0.0
        
        # Cup collision: per-step penalty for gripper touching cup
        if cup_collision:
            components["cup_collision_penalty"] = CUP_COLLISION_PENALTY
        else:
            components["cup_collision_penalty"] = 0.0
        
        # Joint limit penalty: soft penalty for approaching joint limits
        joint_penalty = 0.0
        if joint_positions is not None:
            for idx, name in enumerate(self.env.robot.joint_names):
                lower, upper = self.env.robot.joint_limits[name]
                value = float(joint_positions[idx])
                if value < lower:
                    joint_penalty += lower - value
                elif value > upper:
                    joint_penalty += value - upper
                else:
                    margin = 0.02 if name != "gripper" else 0.002
                    proximity = min(value - lower, upper - value)
                    if proximity < margin:
                        joint_penalty += margin - proximity
        components["joint_limit_penalty"] = -JOINT_LIMIT_PENALTY_WEIGHT * joint_penalty
        
        # Drop penalty: one-time penalty for dropping cube not in cup
        drop_triggered = False
        if self.stage_flags.get("grasped") and not self.stage_flags.get("success"):
            # Cube was grasped but is now on ground and not in cup
            low_height = cube_height is not None and cube_height <= self.env.cube_scale[2] * 0.75
            still_grasping = grasp_detected
            not_in_cup = not in_cup_detected
            if low_height and not still_grasping and not_in_cup:
                drop_triggered = True
        
        if drop_triggered and not self.drop_detected:
            components["drop_penalty"] = DROP_PENALTY
            self.drop_detected = True
        else:
            components["drop_penalty"] = 0.0
        
        # ===== SELF-COLLISION PENALTY =====
        # Penalize gripper approaching base, shoulder, or upper_arm
        # Quadratic penalty: if d < r: penalty = -k * (r - d)^2, else 0
        self_collision_penalty = 0.0
        for link_name in ["base", "shoulder", "upper_arm"]:
            dist = state.get(f"gripper_{link_name}_distance")
            if dist is not None and dist < SELF_COLLISION_THRESHOLD:
                violation = SELF_COLLISION_THRESHOLD - dist
                self_collision_penalty -= SELF_COLLISION_PENALTY_WEIGHT * (violation ** 2)
        components["self_collision_penalty"] = self_collision_penalty
        
        # ===== UPDATE MILESTONE FLAGS =====
        # These are one-way latching flags for PPO tracking
        
        # Reached: gripper within 5cm of cube
        if gripper_cube_distance is not None and gripper_cube_distance < 0.05:
            self.stage_flags["reached"] = True
        
        # Controlled: same as grasped (grasp detected)
        if grasp_detected:
            self.stage_flags["controlled"] = True
        
        # Lifted: cube above ground threshold (3cm)
        if cube_height is not None and cube_height > 0.03:
            self.stage_flags["lifted"] = True
        
        # Above cup: same as droppable range
        if droppable_detected:
            self.stage_flags["above_cup"] = True
        
        # Released: gripper opened after having grasped
        if self._was_grasping and not grasp_detected:
            self.stage_flags["released"] = True
        self._was_grasping = grasp_detected
        
        self.reward_components = components
        return components

    def summarize_reward(self):
        total = float(sum(self.reward_components.values()))
        done = bool(self.stage_flags.get("success"))
        info = {
            "reward_components": self.reward_components.copy(),
            "stage_flags": self.stage_flags.copy(),
            "drop_detected": self.drop_detected,
            # Add debug state
            "task_state": {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                           for k, v in self.task_state.items()},
        }
        validation = getattr(self.env, "last_validation_result", None)
        if validation is not None:
            info["validation"] = validation
        if getattr(self.env, "_force_terminate", False):
            done = True
            info["terminated_by_validation"] = True
            if getattr(self.env, "_termination_reason", None):
                info["termination_reason"] = self.env._termination_reason
        return total, done, info

    def _update_task_state(self):
        env = self.env
        state = {}

        try:
            cube_pos, cube_orient = env.cube.get_world_pose()
        except Exception:
            cube_pos = None
            cube_orient = None

        gripper_pos = None
        gripper_quat = None
        # Prefer the actual gripper link pose captured in the env step.
        if hasattr(env, "_last_gripper_pose"):
            gp, go = env._last_gripper_pose
            gripper_pos = gp
            gripper_quat = go
        if gripper_pos is None:
            wrist_cam = getattr(env.robot, "wrist_camera", None)
            if wrist_cam is not None:
                try:
                    gripper_pos, gripper_quat = wrist_cam.get_world_pose()
                except Exception:
                    gripper_pos, gripper_quat = None, None

        if cube_pos is not None:
            cube_pos = np.array(cube_pos, dtype=np.float32)
        if gripper_pos is not None:
            gripper_pos = np.array(gripper_pos, dtype=np.float32)

        cup_xy = getattr(env, "_cup_xy", None)
        if cup_xy is not None:
            cup_center = np.array([cup_xy[0], cup_xy[1], 0.0], dtype=np.float32)
            cup_top = cup_center + np.array([0.0, 0.0, env.cup_height], dtype=np.float32)
        else:
            cup_center = None
            cup_top = None

        state["cube_pos"] = cube_pos
        state["cube_orient"] = cube_orient
        state["gripper_pos"] = gripper_pos
        state["gripper_orient"] = gripper_quat
        state["cup_center"] = cup_center
        state["cup_top"] = cup_top

        if cube_pos is not None and gripper_pos is not None:
            state["gripper_cube_distance"] = float(np.linalg.norm(gripper_pos - cube_pos))
        else:
            state["gripper_cube_distance"] = None

        if cube_pos is not None:
            state["cube_height"] = float(cube_pos[2])
            if cup_center is not None:
                state["cube_cup_distance_xy"] = float(np.linalg.norm(cube_pos[:2] - cup_center[:2]))
            else:
                state["cube_cup_distance_xy"] = None
            if cup_top is not None:
                state["cube_cup_distance"] = float(np.linalg.norm(cube_pos - cup_top))
            else:
                state["cube_cup_distance"] = None
        else:
            state["cube_height"] = None
            state["cube_cup_distance_xy"] = None
            state["cube_cup_distance"] = None

        state["joint_positions"] = self.latest_joint_positions
        state["joint_velocities"] = self.latest_joint_velocities

        if self.gripper_joint_index is not None and self.latest_joint_positions is not None:
            gripper_value = float(self.latest_joint_positions[self.gripper_joint_index])
        else:
            gripper_value = None
        state["gripper_joint"] = gripper_value
        state["target_gripper"] = self.latest_target_gripper
        
        # Compute physical gripper width as distance between gripper and jaw links
        gripper_link_pos = gripper_pos  # Already computed above
        jaw_link_pos = getattr(env, "_last_jaw_pos", None)
        if gripper_link_pos is not None and jaw_link_pos is not None:
            state["gripper_width"] = float(np.linalg.norm(gripper_link_pos - jaw_link_pos))
        else:
            state["gripper_width"] = None
        
        # Legacy threshold-based closure check (kept for backward compatibility)
        state["gripper_closed"] = gripper_value is not None and gripper_value <= 0.35
        
        # Behavioral grasp detection state
        state["grasp_detected"] = getattr(self.env.gripper_detector, "is_grasping", False)
        state["grasp_position"] = None

        # Self-collision: compute gripper distance to base, shoulder, and upper_arm
        link_positions = {}
        try:
            if hasattr(env, "robot_articulation") and env.robot_articulation is not None:
                link_poses = env._get_link_poses()
                if link_poses and link_poses[0] is not None:
                    positions, _ = link_poses
                    # Link indices: base=0, shoulder=1, upper_arm=2
                    if len(positions) > 0:
                        link_positions["base"] = np.array(positions[0], dtype=np.float32)
                    if len(positions) > 1:
                        link_positions["shoulder"] = np.array(positions[1], dtype=np.float32)
                    if len(positions) > 2:
                        link_positions["upper_arm"] = np.array(positions[2], dtype=np.float32)
        except Exception:
            pass
        
        # Compute distances from gripper to each tracked link
        for link_name, link_pos in link_positions.items():
            if gripper_pos is not None and link_pos is not None:
                state[f"gripper_{link_name}_distance"] = float(np.linalg.norm(gripper_pos - link_pos))
            else:
                state[f"gripper_{link_name}_distance"] = None

        self.task_state = state
