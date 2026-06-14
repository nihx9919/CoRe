import numpy as np
def reward_function(observation, action, target_pos):
    # Extract corner positions from observation
    top_left_pos = observation[:3]
    top_right_pos = observation[15:18]
    bottom_left_pos = observation[90:93]
    bottom_right_pos = observation[105:108]

    # We want to fold diagonally: bring top_left_pos close to bottom_right_pos
    dist_tl_br = np.linalg.norm(np.array(top_left_pos) - np.array(bottom_right_pos))

    # Also, for a diagonal fold, top_right and bottom_left corners should become closer to each other
    dist_tr_bl = np.linalg.norm(np.array(top_right_pos) - np.array(bottom_left_pos))

    # Temperatures for exponential rewards
    temp_dist_tl_br = 0.4
    temp_dist_tr_bl = 0.4

    # Reward components: smaller distance means higher reward
    r_tl_br = np.exp(-dist_tl_br / temp_dist_tl_br) - 0.5  # shifted so min ~ -0.5 max ~0.5
    r_tr_bl = np.exp(-dist_tr_bl / temp_dist_tr_bl) - 0.5

    # Sum components, clamp total reward to [-1, 1]
    total_reward = r_tl_br + r_tr_bl
    total_reward = max(-1, min(1, total_reward))

    individual_rewards = {
        "reward_top_left_to_bottom_right": r_tl_br,
        "reward_top_right_to_bottom_left": r_tr_bl,
    }

    return total_reward, individual_rewards
