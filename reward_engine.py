import numpy as np

REACH_DISTANCE_MAX = 1.0
REACHING_WEIGHT = 1.0
GRASP_BONUS = 2.0
LIFT_BONUS = 2.5
PLACEMENT_WEIGHT = 3.0
PLACEMENT_BONUS = 1.5
SUCCESS_BONUS = 10.0
ACTION_COST_WEIGHT = 0.01
DROP_PENALTY = -3.0
JOINT_LIMIT_PENALTY_WEIGHT = 1.0


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
            "grasped": False,
            "lifted": False,
            "placed": False,
            "success": False,
        }
        self.reward_components = {}
        self.drop_detected = False
        self.task_state = {}
        self.latest_joint_positions = None
        self.latest_joint_velocities = None

    def record_joint_state(self, joint_positions, joint_velocities):
        self.latest_joint_positions = joint_positions
        self.latest_joint_velocities = joint_velocities
        self._update_task_state()

    def compute_reward_components(self):
        state = self.task_state
        components = {}

        gripper_cube_distance = state.get("gripper_cube_distance")
        if gripper_cube_distance is not None:
            reach_ratio = 1.0 - min(gripper_cube_distance, REACH_DISTANCE_MAX) / REACH_DISTANCE_MAX
            reach_ratio = max(0.0, reach_ratio)
            components["reaching"] = REACHING_WEIGHT * reach_ratio
        else:
            components["reaching"] = 0.0

        gripper_closed = state.get("gripper_closed", False)
        cube_height = state.get("cube_height")
        cube_cup_distance = state.get("cube_cup_distance")
        cube_cup_distance_xy = state.get("cube_cup_distance_xy")
        joint_positions = state.get("joint_positions")
        joint_velocities = state.get("joint_velocities")

        # Grasp detection: gripper must be closed AND near cube
        # For 2.5x cube (0.075m), use 4.5x threshold = 0.3375m (~34cm)
        # This accounts for: cube size (7.5cm) + camera offset from gripper (~9cm) + safety margin
        # NOTE: gripper_pos comes from wrist_camera which is offset ~0.09m from actual gripper fingers
        if (not self.stage_flags.get("grasped") and gripper_closed and
                gripper_cube_distance is not None and
                gripper_cube_distance <= self.env.cube_scale[0] * 4.5):
            self.stage_flags["grasped"] = True
            components["grasp_bonus"] = GRASP_BONUS
            print(f"[GRASP] Detected grasp! Distance: {gripper_cube_distance:.3f}m, Gripper value: {state.get('gripper_joint', 0):.3f}")
        else:
            components["grasp_bonus"] = 0.0
            # Debug output when close to grasping
            if gripper_closed and gripper_cube_distance is not None and gripper_cube_distance < 0.4:
                print(f"[GRASP DEBUG] Close but no grasp: dist={gripper_cube_distance:.3f}m, gripper={state.get('gripper_joint', 0):.3f}, closed={gripper_closed}")

        if (self.stage_flags.get("grasped") and not self.stage_flags.get("lifted") and
                cube_height is not None and cube_height >= self.env.cup_height + 0.02):
            self.stage_flags["lifted"] = True
            components["lift_bonus"] = LIFT_BONUS
        else:
            components["lift_bonus"] = 0.0

        placement_progress = 0.0
        if cube_cup_distance is not None:
            placement_progress = 1.0 - min(cube_cup_distance, 0.5) / 0.5
            placement_progress = np.clip(placement_progress, 0.0, 1.0)
        components["placement_shaping"] = PLACEMENT_WEIGHT * placement_progress

        if (self.stage_flags.get("lifted") and not self.stage_flags.get("placed") and
                cube_cup_distance is not None and cube_cup_distance <= 0.05):
            self.stage_flags["placed"] = True
            components["placement_bonus"] = PLACEMENT_BONUS
        else:
            components["placement_bonus"] = 0.0

        gripper_joint = state.get("gripper_joint")
        gripper_open = gripper_joint is not None and gripper_joint >= 0.02
        inside_cup = (
            cube_height is not None and cube_height <= self.env.cup_height and
            cube_cup_distance_xy is not None and cube_cup_distance_xy <= self.env.cup_inner_radius_top * 0.6
        )
        if not self.stage_flags.get("success") and inside_cup and gripper_open:
            self.stage_flags["success"] = True
            components["success_bonus"] = SUCCESS_BONUS
        else:
            components["success_bonus"] = 0.0

        if joint_velocities is not None:
            action_cost = float(np.linalg.norm(joint_velocities))
            components["action_cost"] = -ACTION_COST_WEIGHT * action_cost
        else:
            components["action_cost"] = 0.0

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

        drop_triggered = False
        if self.stage_flags.get("lifted") and not self.stage_flags.get("success"):
            close_to_gripper = (gripper_cube_distance is not None and
                                gripper_cube_distance <= self.env.cube_scale[0] * 1.5)
            low_height = cube_height is not None and cube_height <= self.env.cube_scale[2] * 0.75
            if low_height and (not gripper_closed or not close_to_gripper):
                drop_triggered = True

        if drop_triggered and not self.drop_detected:
            components["drop_penalty"] = DROP_PENALTY
            self.drop_detected = True
        else:
            components["drop_penalty"] = 0.0

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
        # Gripper is "closed" when it's attempting to grasp
        # For 2.5x cube (0.075m), gripper pinches to ~0.12, so use threshold of 0.35
        # Increased threshold to detect closing motion earlier
        state["gripper_closed"] = gripper_value is not None and gripper_value <= 0.35

        self.task_state = state
