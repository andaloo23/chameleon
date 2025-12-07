import numpy as np


def _safe_quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float32)


def _quat_normalize(quat):
    norm = np.linalg.norm(quat)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def _random_quaternion(rng, max_angle_rad):
    axis = rng.normal(size=3)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / axis_norm
    angle = rng.uniform(-max_angle_rad, max_angle_rad)
    half = angle * 0.5
    sin_half = np.sin(half)
    return np.array([axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, np.cos(half)], dtype=np.float32)


class DomainRandomizer:
    def __init__(self, env):
        self.env = env

    def randomize(self):
        # COMPLETELY DISABLED FOR DEBUGGING
        pass
        # self._randomize_lighting()
        # self._randomize_ground()
        # self._randomize_objects()
        # self._randomize_cameras()
        # self._randomize_physics()

    def _randomize_lighting(self):
        stage = self.env.stage_context.get_stage()
        rng = self.env.rng
        try:
            from pxr import UsdLux, Gf
        except ImportError:
            return

        light_path = "/World/RandomLight"
        light_prim = stage.GetPrimAtPath(light_path)
        if not light_prim.IsValid():
            light = UsdLux.DistantLight.Define(stage, light_path)
            light_prim = light.GetPrim()
        light = UsdLux.DistantLight(light_prim)

        intensity = float(rng.uniform(2000.0, 5000.0))
        color = rng.uniform(0.6, 1.0, size=3).astype(float)
        angle = float(rng.uniform(5.0, 20.0))

        light.CreateIntensityAttr().Set(intensity)
        light.CreateColorAttr().Set(Gf.Vec3f(*color))
        light.CreateAngleAttr().Set(angle)

        direction = rng.normal(size=3)
        direction[2] = abs(direction[2]) + 0.2
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        position = direction * float(rng.uniform(6.0, 12.0))
        from pxr import UsdGeom
        UsdGeom.XformCommonAPI(light_prim).SetTranslate(Gf.Vec3d(*position))

    def _randomize_ground(self):
        ground = self.env.world.scene.get_object("GroundPlane")
        if ground is None:
            return

        rng = self.env.rng
        color = rng.uniform(0.2, 0.8, size=3)
        try:
            ground.set_color(color)
        except Exception:
            pass

        try:
            friction = float(rng.uniform(0.7, 1.2))
            physics = self.env.world.get_physics_context()
            physics.set_static_friction(friction)
        except Exception:
            pass

    def _randomize_objects(self):
        rng = self.env.rng

        cube_color = rng.uniform(0.0, 1.0, size=3)
        try:
            self.env.cube.set_color(cube_color)
        except Exception:
            pass

        cup_mesh_path = "/World/Cup/CupMesh"
        try:
            from pxr import UsdGeom, Gf
            stage = self.env.stage_context.get_stage()
            mesh_prim = stage.GetPrimAtPath(cup_mesh_path)
            if mesh_prim.IsValid():
                mesh = UsdGeom.Mesh(mesh_prim)
                cup_color = rng.uniform(0.2, 1.0, size=3)
                mesh.GetDisplayColorAttr().Set([Gf.Vec3f(float(cup_color[0]),
                                                         float(cup_color[1]),
                                                         float(cup_color[2]))])
        except Exception:
            pass

        try:
            scale_noise = rng.uniform(0.95, 1.05, size=3)
            self.env.cube.set_scale(self.env.cube_scale * scale_noise)
        except Exception:
            pass

    def _randomize_cameras(self):
        rng = self.env.rng
        for camera in (self.env.top_camera, self.env.side_camera, getattr(self.env.robot, "wrist_camera", None)):
            if camera is None:
                continue
            try:
                position, orientation = camera.get_world_pose()
            except Exception:
                continue

            position = np.array(position, dtype=np.float32)
            noise_translation = rng.uniform(-0.02, 0.02, size=3)
            if position[2] > 1.0:
                noise_translation[2] = rng.uniform(-0.05, 0.05)
            position = position + noise_translation

            noise_quat = _random_quaternion(rng, max_angle_rad=0.1)
            orientation = np.array(orientation, dtype=np.float32)
            orientation = _quat_normalize(_safe_quat_mul(noise_quat, orientation))

            try:
                camera.set_world_pose(position=position, orientation=orientation)
            except Exception:
                pass

            try:
                exposure = float(rng.uniform(0.8, 1.2))
                gain = float(rng.uniform(0.8, 1.3))
                camera.set_exposure(exposure)
                camera.set_gain(gain)
            except Exception:
                pass

    def _randomize_physics(self):
        rng = self.env.rng
        try:
            physics = self.env.world.get_physics_context()
        except Exception:
            return

        try:
            gravity_scale = float(rng.uniform(0.95, 1.05))
            gravity = np.array([0.0, 0.0, -9.81 * gravity_scale], dtype=float)
            physics.set_gravity(gravity)
        except Exception:
            pass

        try:
            cube_mass = float(rng.uniform(0.15, 0.35))
            self.env.cube.set_mass(cube_mass)
        except Exception:
            pass

        try:
            robot_damping = float(rng.uniform(0.8, 1.2))
            physics.set_default_dof_damping(robot_damping)
        except Exception:
            pass
