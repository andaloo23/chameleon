import os
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.kit.commands import execute as omni_kit_commands_execute
from omni.importer.urdf import _urdf
from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.importer.urdf")

current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

world = World()

urdf_path = os.path.join(current_dir, "so100.urdf")

import_config = _urdf.ImportConfig()
import_config.import_inertia_tensor = True
import_config.fix_base = True
import_config.make_default_prim = True
import_config.create_physics_scene = True
import_config.search_path = [os.path.join(current_dir, "assets")]
import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION

robot_prim_path = "/World/so100_robot"

result, prim_path = omni_kit_commands_execute(
    "URDFParseAndImportFile",
    urdf_path=urdf_path,
    import_config=import_config,
    dest_path=robot_prim_path,
    get_articulation_root=True
)

world.play()
simulation_app.run()
simulation_app.shutdown()