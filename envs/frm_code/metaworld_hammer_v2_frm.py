import numpy as np
def reward_function(observation, action, target_pos):
    # Unpack observation
    # observation = [eef_x, eef_y, eef_z, gripper_open_norm, hammer_x, hammer_y, hammer_z, hammer_quat_w, hammer_quat_x, hammer_quat_y, hammer_quat_z]
    eef_pos = observation[0:3]
    gripper_open = observation[3]  # normalized gripper opening distance
    hammer_pos = observation[4:7]
    hammer_quat = observation[7:11]

    # Compute hammer head position offset relative to hammer base position
    hammer_head_offset = np.array([0.16, 0.06, 0.0])
    hammer_head_pos = hammer_pos + hammer_head_offset

    # --- Reward Components ---

    # 1. Grasp reward: encourage end-effector to be close to hammer handle (hammer base pos) with a nearly closed gripper
    # Distance from eef to hammer base
    dist_eef_to_hammer = np.linalg.norm(eef_pos - hammer_pos)
    # Gripper closing reward: encourage gripper to be closed (gripper_open close to 0)
    gripper_closed = 1 - gripper_open  # 1 when fully closed, 0 when fully open

    # Use exponential decay on distance with temperature for smooth shaping
    temp_grasp = 0.1
    r_eef_close = np.exp(-dist_eef_to_hammer / temp_grasp)

    # Combine with gripper closed signal, penalizing open gripper at hammer handle
    r_grasp = r_eef_close * gripper_closed

    # 2. Hammering reward: distance from hammer head to nail target position
    dist_hammerhead_to_target = np.linalg.norm(hammer_head_pos - target_pos)

    temp_hammer = 0.05
    r_hammer = np.exp(-dist_hammerhead_to_target / temp_hammer)

    # 3. Optional: Small penalty on action magnitude to encourage smooth actions (not mandatory but often useful)
    # We won't include that here since not requested explicitly

    # Aggregate total reward as weighted sum
    # Give more weight to hammering (task success) than just grasping
    # Weights sum to 1 to keep max reward <= 1
    w_grasp = 0.4
    w_hammer = 0.6

    total_reward = w_grasp * r_grasp + w_hammer * r_hammer

    # Since exp outputs are always in [0,1], total_reward in [0,1]
    # We can scale and shift to [-1,1]:  total_reward_scaled = 2*total_reward - 1
    total_reward_scaled = 2 * total_reward - 1

    # Compose component rewards dictionary
    components = {
        "r_grasp": w_grasp * r_grasp,
        "r_hammer": w_hammer * r_hammer,
    }

    # Components sum to total_reward; scale accordingly
    components_scaled = {k: 2*v - (1 if v==0 else 0) for k, v in components.items()}
    # But this linear scaling will break sum to total_reward_scaled; better to keep components unscaled and sum scaled
    # Instead, return components as is (weighted sums), and only scale total reward

    # Return total reward scaled to [-1,1], components as weighted rewards in [0,1]
    return total_reward_scaled, components
