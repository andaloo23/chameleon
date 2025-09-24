from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.urdf import _urdf
import os

world = World()

urdf_path = os.path.join(os.path.dirname(__file__), "so100.urdf")

result, prim_path = _urdf.acquire_urdf_interface().parse_urdf(
    urdf_path=urdf_path,
    import_inertia_tensor=True,
    fix_base=True,
    make_default_prim=True,
    create_physics_scene=True
)

world.reset()
