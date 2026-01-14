import numpy as np

_Gf = None
_UsdGeom = None
_UsdPhysics = None


def initialize_usd_modules(gf_module, usd_geom_module, usd_physics_module):
    global _Gf, _UsdGeom, _UsdPhysics
    _Gf = gf_module
    _UsdGeom = usd_geom_module
    _UsdPhysics = usd_physics_module


def _require_modules():
    if _Gf is None or _UsdGeom is None or _UsdPhysics is None:
        raise RuntimeError("USD modules not initialized. Call initialize_usd_modules first.")


def build_cup_mesh(outer_radius_top, outer_radius_bottom, height,
                   inner_radius_top, inner_radius_bottom,
                   bottom_thickness, segments=32):
    _require_modules()

    bottom_thickness = min(bottom_thickness, height * 0.4)

    points = []
    face_counts = []
    face_indices = []

    def angle(i):
        return (2.0 * np.pi * i) / segments

    def add_point(x, y, z):
        points.append(_Gf.Vec3f(float(x), float(y), float(z)))

    for i in range(segments):
        ang = angle(i)
        add_point(outer_radius_bottom * np.cos(ang), outer_radius_bottom * np.sin(ang), 0.0)
    outer_top_offset = len(points)
    for i in range(segments):
        ang = angle(i)
        add_point(outer_radius_top * np.cos(ang), outer_radius_top * np.sin(ang), height)
    inner_top_offset = len(points)
    for i in range(segments):
        ang = angle(i)
        add_point(inner_radius_top * np.cos(ang), inner_radius_top * np.sin(ang), height)
    inner_bottom_offset = len(points)
    for i in range(segments):
        ang = angle(i)
        add_point(inner_radius_bottom * np.cos(ang), inner_radius_bottom * np.sin(ang), bottom_thickness)

    bottom_center_top_idx = len(points)
    add_point(0.0, 0.0, bottom_thickness)
    bottom_center_bottom_idx = len(points)
    add_point(0.0, 0.0, 0.0)

    def add_triangle(a, b, c):
        face_counts.append(3)
        face_indices.extend([a, b, c])

    segments_range = range(segments)
    for i in segments_range:
        next_i = (i + 1) % segments
        ob_i = i
        ob_next = next_i
        ot_i = outer_top_offset + i
        ot_next = outer_top_offset + next_i
        add_triangle(ob_i, ob_next, ot_i)
        add_triangle(ot_i, ob_next, ot_next)

    for i in segments_range:
        next_i = (i + 1) % segments
        ib_i = inner_bottom_offset + i
        ib_next = inner_bottom_offset + next_i
        it_i = inner_top_offset + i
        it_next = inner_top_offset + next_i
        add_triangle(ib_i, it_i, ib_next)
        add_triangle(it_i, it_next, ib_next)

    for i in segments_range:
        next_i = (i + 1) % segments
        ob_i = i
        ob_next = next_i
        ib_i = inner_bottom_offset + i
        ib_next = inner_bottom_offset + next_i
        add_triangle(ob_i, ib_i, ob_next)
        add_triangle(ob_next, ib_i, ib_next)

    for i in segments_range:
        next_i = (i + 1) % segments
        ib_i = inner_bottom_offset + i
        ib_next = inner_bottom_offset + next_i
        add_triangle(ib_i, ib_next, bottom_center_top_idx)

    for i in segments_range:
        next_i = (i + 1) % segments
        ob_i = i
        ob_next = next_i
        add_triangle(ob_i, bottom_center_bottom_idx, ob_next)

    # Added: Connecting outer top ring to inner top ring to create a "thickness" rim
    for i in segments_range:
        next_i = (i + 1) % segments
        ot_i = outer_top_offset + i
        ot_next = outer_top_offset + next_i
        it_i = inner_top_offset + i
        it_next = inner_top_offset + next_i
        # Triangle 1
        add_triangle(ot_i, it_i, ot_next)
        # Triangle 2
        add_triangle(it_i, it_next, ot_next)

    return points, face_counts, face_indices


def create_cup_prim(stage, prim_path, position,
                    outer_radius_top, outer_radius_bottom,
                    inner_radius_top, inner_radius_bottom,
                    height, bottom_thickness, color, mass):
    _require_modules()

    xform = _UsdGeom.Xform.Define(stage, prim_path)
    _UsdGeom.XformCommonAPI(xform).SetTranslate(
        _Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    )

    mesh_path = f"{prim_path}/CupMesh"
    mesh = _UsdGeom.Mesh.Define(stage, mesh_path)
    points, counts, indices = build_cup_mesh(
        outer_radius_top, outer_radius_bottom, height,
        inner_radius_top, inner_radius_bottom, bottom_thickness
    )
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateDisplayColorAttr().Set([
        _Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
    ])

    _UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    _UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("convexDecomposition")
    xform_prim = xform.GetPrim()
    rigid_api = _UsdPhysics.RigidBodyAPI.Apply(xform_prim)
    rigid_api.CreateRigidBodyEnabledAttr(True)
    # Make cup kinematic - it won't be pushed around and won't trap the gripper
    rigid_api.CreateKinematicEnabledAttr(True)
    mass_api = _UsdPhysics.MassAPI.Apply(xform_prim)
    mass_attr = mass_api.GetMassAttr()
    if not mass_attr:
        mass_attr = mass_api.CreateMassAttr()
    mass_attr.Set(float(mass))
    return xform
