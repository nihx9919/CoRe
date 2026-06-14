import numpy as np
def reward_function(observation, action, target_pos):
    """
    Reward function to move the ball into the goal with the robot end-effector.

    observation: np.array of shape (11,)
        [end_effector_x, end_effector_y, end_effector_z,
         gripper_open, 
         ball_x, ball_y, ball_z,
         ball_quat_w, ball_quat_x, ball_quat_y, ball_quat_z]
    action: np.array of shape (4,)
        [delta_x, delta_y, delta_z, gripper_torque]
    target_pos: np.array of shape (3,)
        Target xyz position of the ball (goal position)
    """

    # Extract positions
    end_effector_pos = observation[0:3]
    ball_pos = observation[4:7]

    # Distance ball to goal
    ball_to_goal_dist = np.linalg.norm(ball_pos - target_pos)
    # Distance end-effector to ball
    ee_to_ball_dist = np.linalg.norm(end_effector_pos - ball_pos)

    # Temperature parameters for smoothing
    temp_ball_goal = 5.0
    temp_ee_ball = 5.0

    # Reward component 1: Encourage ball close to goal (exponentially scaled negative distance)
    r_ball_goal = np.exp(-temp_ball_goal * ball_to_goal_dist) - 1  # range (-1, 0], closer = values closer to 0

    # Reward component 2: Encourage end-effector close to ball to promote interaction
    r_ee_ball = np.exp(-temp_ee_ball * ee_to_ball_dist) - 1  # range (-1, 0]

    # Combine, weighted sum (weights sum to 1)
    w_ball_goal = 0.7
    w_ee_ball = 0.3
    reward = w_ball_goal * r_ball_goal + w_ee_ball * r_ee_ball  # in [-1,0]

    # Rescale reward to [-1,1] so maximum is near 0 -> shift and scale
    # max r_ball_goal ~0, min ~-1; same for r_ee_ball
    # So minimal reward close to -1, max at 0
    # Shift rewards by +1 to [0,1], then weighted sum in [0,1], then scale to [-1,1]
    reward_shifted = w_ball_goal * (r_ball_goal + 1) + w_ee_ball * (r_ee_ball + 1)  # in [0,1]
    reward_final = 2 * reward_shifted - 1  # in [-1,1]

    # Individual components reported after shifting to be consistent with sum = total reward
    r_ball_goal_shifted = w_ball_goal * (r_ball_goal + 1)
    r_ee_ball_shifted = w_ee_ball * (r_ee_ball + 1)

    individual_rewards = {
        "ball_to_goal": r_ball_goal_shifted,
        "ee_to_ball": r_ee_ball_shifted,
    }

    return reward_final, individual_rewards
