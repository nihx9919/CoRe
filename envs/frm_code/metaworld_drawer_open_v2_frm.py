import numpy as np
def reward_function(observation, action, target_pos):
    # Extract relevant components from observation
    # observation = [pos_hand_xyz(3), gripper_open_norm(1), first_obj_pos_xyz(3), first_obj_quat(4)]
    pos_hand = observation[0:3]         # End-effector position (x,y,z)
    gripper_open = observation[3]       # Gripper opening, normalized (0 closed, 1 open)
    first_obj_pos = observation[4:7]    # Drawer handle position (x,y,z)
    # first_obj_quat = observation[7:11]  # Not used directly here

    # Compute distance from end-effector to drawer handle (encourage close interaction)
    dist_hand_to_handle = np.linalg.norm(pos_hand - first_obj_pos)
    # Compute distance from drawer handle to target position (encourage drawer opening)
    dist_handle_to_target = np.linalg.norm(first_obj_pos - target_pos)

    # Temperatures for exponential scaling
    temp_hand_dist = 0.5
    temp_handle_target = 0.3
    temp_gripper = 5.0

    # Reward for end-effector being close to the handle (higher when closer)
    r_hand_close = np.exp(-dist_hand_to_handle / temp_hand_dist)  # in (0,1]

    # Reward for drawer handle moving close to the target (higher when closer)
    r_handle_target = np.exp(-dist_handle_to_target / temp_handle_target)

    # Reward encouraging the gripper to close (to pull handle and open drawer)
    # Note: gripper_open is normalized (0 closed, 1 open). We want it to be closed to pull drawer.
    r_gripper_closed = np.exp(- (gripper_open) * temp_gripper)  # close = 0 => reward ~1, open=1 => reward ~exp(-5) ~0.0067

    # Compose final reward as weighted sum
    # Emphasize drawer opening (r_handle_target) most, then gripper close, then hand close
    reward = 0.5 * r_handle_target + 0.3 * r_gripper_closed + 0.2 * r_hand_close

    # Shift reward range [0,1] to [-1,1]
    total_reward = 2 * reward - 1

    # Clip total reward to ensure within [-1,1]
    total_reward = np.clip(total_reward, -1, 1)

    # Sum of components must equal total reward, so re-scale components accordingly
    sum_components = r_handle_target + r_gripper_closed + r_hand_close
    r_handle_target_scaled = 0.5 * r_handle_target
    r_gripper_closed_scaled = 0.3 * r_gripper_closed
    r_hand_close_scaled = 0.2 * r_hand_close
    sum_scaled = r_handle_target_scaled + r_gripper_closed_scaled + r_hand_close_scaled
    # Normalize to match total_reward linearly (avoid floating issue)
    scaling_factor = (total_reward + 1) / (2 * sum_scaled) if sum_scaled > 0 else 0
    r_handle_target_final = r_handle_target_scaled * scaling_factor
    r_gripper_closed_final = r_gripper_closed_scaled * scaling_factor
    r_hand_close_final = r_hand_close_scaled * scaling_factor

    reward_components = {
        "handle_to_target": r_handle_target_final,
        "gripper_closed": r_gripper_closed_final,
        "hand_to_handle": r_hand_close_final,
    }

    # Sum components to form total reward (should equal total_reward)
    final_sum = r_handle_target_final + r_gripper_closed_final + r_hand_close_final
    # In rare cases of zero sum_scaled, fallback to total_reward directly
    if abs(final_sum - total_reward) > 1e-6:
        reward_components = {
            "handle_to_target": total_reward / 3,
            "gripper_closed": total_reward / 3,
            "hand_to_handle": total_reward / 3,
        }
        total_reward = sum(reward_components.values())

    return total_reward, reward_components
