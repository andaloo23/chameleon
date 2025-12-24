import numpy as np

def forward_kinematics(shoulder_lift, elbow_flex, wrist_flex):
    # Link lengths (m)
    L1 = 0.1160
    L2 = 0.1350
    L3 = 0.0601
    
    # Shoulder pivot (from my derivation)
    R0 = 0.0758
    Z0 = 0.119
    
    # Angle offsets (Mapping finding)
    # Based on URDF: shoulder_lift=2.0, elbow_flex=-1.0, wrist_flex=-0.5 is a plausible pose.
    # We need to map [shoulder_lift, elbow_flex, wrist_flex] to geometric angles [th1, th2, th3]
    # th1: angle of L1 relative to horizontal
    # th2: angle of L2 relative to L1
    # th3: angle of L3 relative to L2
    
    # Let's assume th1 = 2.1 - shoulder_lift? No.
    # Let's assume th2 = elbow_flex?
    # th1 = shoulder_lift - 2.1?
    
    # Let's try to match the geometric th1, th2, th3
    # r = R0 + L1*cos(th1) + L2*cos(th1+th2) + L3*cos(th1+th2+th3)
    # z = Z0 + L1*sin(th1) + L2*sin(th1+th2) + L3*sin(th1+th2+th3)
    
    # Mapping based on common sense: 
    th1 = shoulder_lift - 2.1
    th2 = elbow_flex
    th3 = wrist_flex + 0.8  # Wrist flex is relative
    
    r = R0 + L1 * np.cos(th1) + L2 * np.cos(th1 + th2) + L3 * np.cos(th1 + th2 + th3)
    z = Z0 + L1 * np.sin(th1) + L2 * np.sin(th1 + th2) + L3 * np.sin(th1 + th2 + th3)
    
    return r, z

def inverse_kinematics(target_r, target_z, phi=-0.6):
    # Link lengths (mm)
    L1 = 0.1160
    L2 = 0.1350
    L3 = 0.0601
    R0 = 0.0758
    Z0 = 0.119
    
    r_rel = target_r - R0
    z_rel = target_z - Z0
    
    # Wrist pivot in R-Z plane
    r_w = r_rel - L3 * np.cos(phi)
    z_w = z_rel - L3 * np.sin(phi)
    
    dist_sq = r_w**2 + z_w**2
    dist = np.sqrt(dist_sq)
    
    if dist > (L1 + L2): return None
    
    cos_elbow = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)
    th2 = -np.arccos(np.clip(cos_elbow, -1, 1))
    
    alpha = np.arctan2(z_w, r_w)
    beta = np.arctan2(L2 * np.sin(th2), L1 + L2 * np.cos(th2))
    th1 = alpha - beta
    
    # Joint mapping
    shoulder_lift = th1 + 2.1
    elbow_flex = th2
    wrist_flex = phi - (th1 + th2) - 0.8
    
    return shoulder_lift, elbow_flex, wrist_flex

if __name__ == "__main__":
    # Test target: r=0.32, z=0.038
    print("Testing r=0.32, z=0.038")
    sol = inverse_kinematics(0.32, 0.038)
    if sol:
        s, e, w = sol
        print(f"Sol: s={s:.3f}, e={e:.3f}, w={w:.3f}")
        r_rv, z_rv = forward_kinematics(s, e, w)
        print(f"Verify: r={r_rv:.3f}, z={z_rv:.3f}")
    
    # Test target closer: r=0.25 (should fail or be very tucked)
    print("\nTesting r=0.25, z=0.038")
    sol = inverse_kinematics(0.25, 0.038)
    if sol:
        s, e, w = sol
        print(f"Sol: s={s:.3f}, e={e:.3f}, w={w:.3f}")
        r_rv, z_rv = forward_kinematics(s, e, w)
        print(f"Verify: r={r_rv:.3f}, z={z_rv:.3f}")

    # Max reach check
    print(f"\nMax reach from base: {0.0758 + 0.116 + 0.135 + 0.060:.3f}m")
