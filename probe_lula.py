import lula
import yaml
import os

print(f"Lula version: {getattr(lula, '__version__', 'unknown')}")
print(f"Lula dir: {dir(lula)}")

# Try to look for any default configs or schema info in the library path
try:
    import inspect
    print(f"Lula file: {inspect.getfile(lula)}")
except:
    pass

# Try creating a minimal RmpFlow policy to see what it says
try:
    # We need a URDF and robot description. Let's use the ones we have.
    # Note: this might fail if we don't have a CUDA context, but worth a shot.
    # Actually, lula usually doesn't need CUDA for just parsing.
    # But RmpFlow might.
    print("\nAttempting to parse current configs...")
    with open("robot_description.yaml", 'r') as f:
        rd = yaml.safe_load(f)
    with open("rmpflow_config.yaml", 'r') as f:
        rc = yaml.safe_load(f)
    
    print("Robot description keys:", list(rd.keys()) if rd else "None")
    print("RmpFlow config keys:", list(rc.keys()) if rc else "None")
except Exception as e:
    print(f"Error during probe: {e}")
