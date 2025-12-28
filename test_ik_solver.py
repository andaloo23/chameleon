import numpy as np

def forward_kinematics(q):
    """
    Forward kinematics for SO-100.
    """
    pan, lift, elbow, wrist = q
    
    # Correctly scaled link lengths (URDF values * 2.5)
    L1 = 0.1160 * 2.5  # Upper arm
    L2 = 0.1350 * 2.5  # Lower arm
    
    # L3 includes the wrist-to-gripper offset (0.06m) PLUS jaw length (~0.1m)
    # Total unscaled L3 ~ 0.16m
    L3 = 0.1600 * 2.5  
    
    R0, Z0 = 0.0758 * 2.5, 0.119 * 2.5
    
    th1 = lift - 0.228
    th2_rel = elbow + 1.571
    th3_rel = wrist - 1.0
    
    r = R0 + L1 * np.cos(th1) + L2 * np.cos(th1 + th2_rel) + L3 * np.cos(th1 + th2_rel + th3_rel)
    z = Z0 + L1 * np.sin(th1) + L2 * np.sin(th1 + th2_rel) + L3 * np.sin(th1 + th2_rel + th3_rel)
    
    return r, z

def inverse_kinematics(target_r, target_z):
    """
    Optimized Inverse kinematics for SO-100.
    """
    L1 = 0.1160 * 2.5
    L2 = 0.1350 * 2.5
    L3 = 0.1600 * 2.5
    
    R0, Z0 = 0.0758 * 2.5, 0.119 * 2.5
    
    r_rel = target_r - R0
    z_rel = target_z - Z0
    
    limits = {
        "lift": (0.0, 3.5),
        "elbow": (-3.14158, 0.0),
        "wrist": (-2.5, 1.2)
    }
    
    best_score = -float('inf')
    best_sol = None
    
    # Search for optimal phi (total angle of gripper relative to horizontal)
    for phi in np.linspace(-np.pi, 0.5 * np.pi, 100):
        r_w = r_rel - L3 * np.cos(phi)
        z_w = z_rel - L3 * np.sin(phi)
        
        dist_sq = r_w**2 + z_w**2
        dist = np.sqrt(dist_sq)
        
        if dist > (L1 + L2) or dist < abs(L1 - L2):
            continue
        
        for elbow_sign in [-1, 1]:
            cos_elbow = (dist_sq - L1**2 - L2**2) / (2 * L1 * L2)
            th2_rel = elbow_sign * np.arccos(np.clip(cos_elbow, -1, 1))
            
            alpha = np.arctan2(z_w, r_w)
            beta = np.arctan2(L2 * np.sin(th2_rel), L1 + L2 * np.cos(th2_rel))
            th1 = alpha - beta
            
            s_lift = th1 + 0.228
            e_flex = th2_rel - 1.571
            w_flex = phi - (th1 + th2_rel) + 1.0
            
            if not (limits["lift"][0] <= s_lift <= limits["lift"][1] and
                    limits["elbow"][0] <= e_flex <= limits["elbow"][1] and
                    limits["wrist"][0] <= w_flex <= limits["wrist"][1]):
                continue
                
            # Scoring: 
            # 1. Prefer shoulder to be LOWER (smaller s_lift)
            # 2. Prefer elbow to be "up" configurations if possible
            # 3. Prefer mid-range for joints
            score = -s_lift
            
            for val, lim in [(s_lift, limits["lift"]), (e_flex, limits["elbow"]), (w_flex, limits["wrist"])]:
                center = (lim[0] + lim[1]) / 2.0
                span = lim[1] - lim[0]
                score -= 0.5 * (abs(val - center) / span)**2
            
            # Prefer downward approach for grasping
            score -= 0.5 * abs(phi + 1.2) 
            
            if score > best_score:
                best_score = score
                best_sol = (s_lift, e_flex, w_flex, phi)
                
    return best_sol

if __name__ == "__main__":
    targets = [
        (1.0, 0.1, "Extreme reach"),
        (0.8, 0.1, "Far reach"),
        (0.4, 0.038, "Close reach"),
        (0.6, 0.2, "Medium height"),
        (0.3, 0.4, "High close"),
    ]
    
    for r, z, label in targets:
        print(f"\n--- Testing {label}: r={r}, z={z} ---")
        sol = inverse_kinematics(r, z)
        if sol:
            s, e, w, phi = sol
            print(f"Optimal phi: {phi:.3f} ({np.degrees(phi):.1f} deg)")
            print(f"Sol: s_lift={s:.3f}, e_flex={e:.3f}, w_flex={w:.3f}")
            r_rv, z_rv = forward_kinematics([0.0, s, e, w])
            print(f"Verify: r={r_rv:.3f}, z={z_rv:.3f}")
            print(f"Error: {np.sqrt((r_rv-r)**2 + (z_rv-z)**2):.6f}")
        else:
            print("No solution found")
    
    L1, L2, L3 = 0.1160 * 2.5, 0.1350 * 2.5, 0.1600 * 2.5
    R0 = 0.0758 * 2.5
    print(f"\nMax theoretical reach: {R0 + L1 + L2 + L3:.3f}m")
