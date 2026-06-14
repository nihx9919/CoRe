class:
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        # The observation space is represented as the 3D Cartesian coordinates of particles that make up a flexible object.
        # In this task, end_point1, end_point2 respectively represent the current positions of the two endpoints of the rope.
        # The rope is laid flat in the horizontal plane represented by the X-axis and the Z-axis.
        end_point1, end_point2 = observation[:3], observation[27:30]
        # The action space is the changes in 3D space of the two endpoints of the rope. The actions in this space range between −0.01 and 0.01.
        end_point1_delta_x, end_point1_delta_y, end_point1_delta_z = action[:3]
        end_point1_delta_x, end_point1_delta_y, end_point1_delta_z = action[4:7]
        # Please note that target_pos is not required for this task, and action can be used or not.

