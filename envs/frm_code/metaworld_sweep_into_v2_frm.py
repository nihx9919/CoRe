import numpy as np
def reward_function(observation, action, target_pos):
    # Unpack observation components
    # observation = [x_hand, y_hand, z_hand, gripper_open, x_obj, y_obj, z_obj, qx_obj, qy_obj, qz_obj, qw_obj]
    hand_pos = observation[0:3]
    # gripper_open = observation[3]  # Normalized gripper opening, not needed for sweep task specifically
    obj_pos = observation[4:7]
    # obj_quat = observation[7:11]   # Not used in reward for sweeping
    
    # Distance from object to target (hole)
    dist_obj_to_target = np.linalg.norm(obj_pos - target_pos)
    
    # Distance from hand to object
    dist_hand_to_obj = np.linalg.norm(hand_pos - obj_pos)
    
    # We want to reward the cube being closer to the hole,
    # and encourage the hand to stay near the cube to sweep it.
    # Also encourage progress towards the goal.
    
    # Temperature parameters for exponential scaling
    temp_obj_to_target = 0.5
    temp_hand_to_obj = 1.0
    
    # Reward component: cube closer to target gives higher reward (range 0 to 1)
    # Apply negative distance scaled and exponentiated
    r_obj_to_target = np.exp(-dist_obj_to_target / temp_obj_to_target) - 0.5  # shift to roughly center near 0
    
    # Reward component: hand close to object (to encourage interaction)
    r_hand_to_obj = np.exp(-dist_hand_to_obj / temp_hand_to_obj) - 0.5  # shift to roughly center near 0
    
    # Combine rewards with weights
    # Since these scores range roughly in [-0.5, 0.5], sum ranges [-1, 1]
    reward = r_obj_to_target + r_hand_to_obj
    
    # Clip total reward to [-1, 1]
    reward = np.clip(reward, -1.0, 1.0)
    
    # Compose individual reward dict for info/debugging
    reward_components = {
        "cube_to_target": r_obj_to_target,
        "hand_to_cube": r_hand_to_obj,
    }
    
    return reward, reward_components