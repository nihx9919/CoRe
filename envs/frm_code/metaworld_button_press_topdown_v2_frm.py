import numpy as np
def reward_function(observation, action, target_pos):
    # Extract information from observation
    # observation = [eef_x, eef_y, eef_z, gripper_open, obj_x, obj_y, obj_z, obj_qw, obj_qx, obj_qy, obj_qz]
    eef_pos = observation[0:3]
    gripper_open = observation[3]
    obj_pos = observation[4:7]
    obj_quat = observation[7:11]

    # Reward components:
    # 1) Encourage the button (object) to move down (along z-axis) towards or below target_pos z coordinate
    #    The button's z coordinate lower than initial means pressed down.
    # 2) Encourage the end-effector to be near the button to facilitate pressing
    # 3) Encourage the gripper torque to apply pressing force (optional, based on action)
    
    # Temperature parameters for smooth exponential transforms
    temp_press = 5.0
    temp_dist = 5.0

    # Button pressed reward:
    # The target is to press the button down: reward based on how far the button is pushed
    # The lower the button (z axis), the higher the reward. We clamp relative to initial target z (the nominal "top" position).
    # If button below or equal to target_pos[2] (pressed fully down), max reward 1.
    button_press_depth = target_pos[2] - obj_pos[2]  # positive means button pressed down
    button_press_depth = max(button_press_depth, 0)  # ignore if pressed above target pos z
    # Normalize by a typical max press depth (assumed 0.04m = 4cm)
    max_press_depth = 0.04
    norm_press = np.clip(button_press_depth / max_press_depth, 0, 1)
    r_press = np.tanh(norm_press * temp_press)  # smooth sharp increase to 1

    # Distance from end-effector to button position
    dist_eef_obj = np.linalg.norm(eef_pos - obj_pos)
    dist_norm = np.clip(dist_eef_obj / 0.2, 0, 1)  # assuming 20cm max relevant range
    r_dist = np.tanh((1 - dist_norm) * temp_dist)  # reward closer proximity

    # Penalize gripper being open (encourage slight closing, assuming gripper_open ∈ [0,1], 0 = fully closed)
    r_gripper = -gripper_open  # reward closing gripper for pressing

    # Compose total reward from components with weights summing to 1:
    # Pressing is the main goal, proximity supports pressing, closing the gripper helps
    w_press = 0.7
    w_dist = 0.2
    w_grip = 0.1

    total_reward = w_press * r_press + w_dist * r_dist + w_grip * r_gripper
    # Clip total reward to [-1, 1]
    total_reward = np.clip(total_reward, -1, 1)

    return total_reward, {
        "press_reward": r_press,
        "distance_reward": r_dist,
        "gripper_reward": r_gripper,
    }
