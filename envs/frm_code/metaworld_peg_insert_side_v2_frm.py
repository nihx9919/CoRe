import numpy as np
def reward_function(observation, action, target_pos):
    # Unpack observation components based on given structure
    # observation: [x_eef, y_eef, z_eef, gripper_open_norm, x_obj, y_obj, z_obj, qx_obj, qy_obj, qz_obj, qw_obj]
    eef_pos = np.array(observation[0:3])
    gripper_open = observation[3]  # normalized [0,1], 0 means closed, 1 means fully open
    obj_pos = np.array(observation[4:7])
    obj_quat = np.array(observation[7:11])
    
    # Compute peg_head_position relative to first_obj_pos (peg)
    peg_head_pos = obj_pos + np.array([-0.12, 0.0, 0.0])
    
    ## Reward components:
    # 1) Distance from peg_head to hole (target_pos) - encourage insertion
    dist_peg_to_hole = np.linalg.norm(peg_head_pos - target_pos)
    temp_peg = 10.0
    r_peg_to_hole = np.exp(-temp_peg * dist_peg_to_hole)  # in (0,1], closer => higher reward
    
    # 2) Distance from eef to peg_head - encourage approaching peg for grasping
    dist_eef_peg = np.linalg.norm(eef_pos - peg_head_pos)
    temp_eef = 10.0
    r_eef_peg = np.exp(-temp_eef * dist_eef_peg)  # higher when close
    
    # 3) Gripper closed to grasp (small gripper_open value means closed)
    # We want gripper to be roughly closed when near peg for grasp, but not to excessively penalize fully closed or fully open.
    # Use 1 - gripper_open so closed ~1 reward, open ~0.
    r_grasp = 1 - gripper_open  # in [0,1], larger means more closed
    
    # 4) Bonus for successful insertion: peg_head close enough to hole and gripper closed (i.e. grasp done)
    inserted_threshold = 0.02
    grasped_threshold = 0.1  # gripper fairly closed
    inserted = float(dist_peg_to_hole < inserted_threshold and gripper_open < grasped_threshold)
    
    # Compose total reward as weighted sum, then clamp sum to [-1,1]
    # We weigh insertion highest, then closeness of peg to hole, approach of eef to peg and finally grasping.
    w_inserted = 0.5
    w_peg_to_hole = 0.3
    w_eef_peg = 0.15
    w_grasp = 0.05
    
    reward = (
        w_inserted * inserted +
        w_peg_to_hole * r_peg_to_hole +
        w_eef_peg * r_eef_peg +
        w_grasp * r_grasp
    )
    
    # Normalize reward to max 1 (since inserted can be 0.5 max, sum of weights is 1)
    # Lower bound could be 0 if all components zero, map to range [-1,1]:
    # Let's linearly scale reward from [0,1] to [-1,1] to use full range.
    reward = 2 * reward - 1
    reward = np.clip(reward, -1, 1)
    
    components = {
        "reward_inserted": w_inserted * inserted * 2 - w_inserted,  # scaled similarly
        "reward_peg_to_hole": w_peg_to_hole * (r_peg_to_hole * 2 - 1),
        "reward_eef_to_peg": w_eef_peg * (r_eef_peg * 2 - 1),
        "reward_grasp": w_grasp * (r_grasp * 2 - 1)
    }
    
    # Sum of components should equal total reward, but we scaled components separately. 
    # To guarantee sum equals reward, we re-scale components accordingly:
    comp_sum = sum(components.values())
    if comp_sum != 0:
        scale = reward / comp_sum
    else:
        scale = 0
    for k in components:
        components[k] *= scale
        
    return reward, components