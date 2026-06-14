import numpy as np
def reward_function(observation, action, target_pos):
    # Extract endpoints positions
    end_point1 = observation[:3]
    end_point2 = observation[27:30]

    # Vector from end_point1 to end_point2
    vec = end_point2 - end_point1

    # Temperature parameters for transformations
    temp_dist = 0.1  # for distance from straight line (normalized)
    temp_straight = 0.5  # for straightness score

    # Compute the rope points positions in 3D, assuming observations between endpoints correspond to inner points of rope
    # We focus on the 28 points from obs[3:27] in 3D (i.e. 24 elements x 3 coords = 72 ?)
    # Correction: Observation length and indexing are partial. Typically, 30 observations imply 10 particles (10*3=30).
    # Given endpoints at obs[0:3], and obs[27:30], the middle points are from obs[3:27].
    # So middle 8 points = 8*3=24 elements at obs[3:27].

    points = np.reshape(observation[3:27], (-1, 3))  # 8 middle points shape (8,3)

    # For each middle point, compute its distance from the line defined by end_point1 and end_point2
    # Distance from point p to line defined by points a,b is norm(cross(b - a, p - a)) / norm(b - a)
    line_vec = vec
    line_length = np.linalg.norm(line_vec)
    if line_length < 1e-8:
        # If endpoints coincide, no rope length, reward zero
        return 0.0, {"reward_line_dist": 0.0, "reward_length": 0.0}

    line_dir = line_vec / line_length

    dists = []
    for p in points:
        ap = p - end_point1
        cross_prod = np.cross(line_dir, ap)
        dist = np.linalg.norm(cross_prod)
        dists.append(dist)
    dists = np.array(dists)

    # Compute average distance of inner points from line (smaller is straighter)
    avg_dist = np.mean(dists)

    # Reward component 1: encourage minimizing the average distance from line (rope straightness)
    # We apply exponential decay to scale reward to [0,1] with temp_dist controlling sharpness
    reward_line_dist = np.exp(-avg_dist / temp_dist) - 1  # will be in (-1,0], closer to 0 better

    # Reward component 2: encourage longer rope length (distance between endpoints)
    # Max rope length is unknown; normalize by a heuristic max length, say 1.5 (can be tuned)
    max_length = 1.5
    norm_length = min(line_length / max_length, 1.0)  # in [0, 1]
    # Apply exponential scaling to emphasize reaching max length
    reward_length = np.exp(norm_length / temp_straight) - 1  # in (-1, ~1) but capped below by clamp

    # Combine rewards
    total_reward = reward_line_dist + reward_length

    # Normalize total reward to be in [-1,1]
    total_reward = max(min(total_reward, 1.0), -1.0)

    return total_reward, {
        "reward_line_dist": reward_line_dist,
        "reward_length": reward_length
    }
