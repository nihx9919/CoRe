import numpy as np
def reward_function(observation, action, target_pos):
    # Extract components from observation
    # observation: [x_eef, y_eef, z_eef, gripper_open_norm, x_obj, y_obj, z_obj]
    pos_eef = observation[0:3]
    gripper_open = observation[3]
    obj_pos = observation[4:7]

    # Temperature parameters for exponential transformations
    temp_pos = 0.5
    temp_grip = 0.8

    # Distance from end-effector to dial (encourages proximity to dial)
    dist_eef_obj = np.linalg.norm(pos_eef - obj_pos)
    r_pos = np.exp(-temp_pos * dist_eef_obj)  # in (0,1]

    # Distance from dial's position to target position (encourages dial turning)
    dist_obj_target = np.linalg.norm(obj_pos - target_pos)
    r_target = 1 - np.exp(-temp_pos * dist_obj_target)  # dial closer to target should yield higher reward, so invert

    # Encourage gripper torque application (action[3]) to be moderate (not fully closed or open)
    # Since turning the dial often requires the gripper to hold but not fully close
    grip_torque = action[3]
    r_grip = 1 - abs(grip_torque)  # reward close to zero torque moderate grip, scaled after

    # Combine components:
    # - Encourage end effector close to dial (r_pos)
    # - Encourage dial close to target (r_target)
    # - Moderate torque (r_grip)
    # We sum r_pos + r_target + a scaled r_grip (-ve effect to penalize extremes)
    # Normalize sum to [-1,1] by scaling properly

    # Weighting components
    w_pos = 0.4
    w_target = 0.5
    w_grip = 0.1

    reward = w_pos * (2*r_pos -1) + w_target * (2*r_target -1) + w_grip * (2*r_grip -1)

    # Clamp reward to [-1,1]
    reward = max(min(reward, 1), -1)

    reward_components = {
        "r_pos": w_pos * (2*r_pos -1),
        "r_target": w_target * (2*r_target -1),
        "r_grip": w_grip * (2*r_grip -1)
    }

    # Sum components for consistency check
    total_components = sum(reward_components.values())
    # Minor float difference may exist, adjust reward to exactly component sum
    reward = total_components

    return reward, reward_components
