import numpy as np
def reward_function(observation, action, target_pos):
    # Unpack observation variables
    cup_pos_x, cup_length, cup_width, cup_height, water_height, water_in_cup, water_out_of_cup = observation
    
    # Parameters for reward shaping temperatures
    dist_temp = 0.5
    spill_temp = 5.0
    
    # Compute distance of the cup center to the target position (red circle)
    dist = abs(cup_pos_x - target_pos)
    # Normalize distance by the max possible range assuming the environment's range is about [-1,1]
    dist_norm = dist / 2.0  # assuming the max range is roughly 2 units
    
    # Distance reward: exponential decay on the distance (closer is better)
    distance_reward = np.exp(-dist_temp * dist_norm)
    
    # Spill penalty: exponential decay on water spilled (less spill is better)
    # water_out_of_cup is the proportion of water spilled, range [0,1]
    spill_penalty = -np.exp(spill_temp * water_out_of_cup) + 1  # shifted to be 0 when no spill
    
    # Combine rewards: distance_reward in [0,1], spill_penalty in [-1,0]
    # Total reward in [-1, 1]
    total_reward = distance_reward + spill_penalty
    # Clip to [-1, 1] for safety
    total_reward = np.clip(total_reward, -1, 1)
    
    individual_rewards = {
        'distance_reward': distance_reward,
        'spill_penalty': spill_penalty
    }
    
    return total_reward, individual_rewards