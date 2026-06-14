class:
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        # The observation space is represented as the 3D Cartesian coordinates of particles that make up a flexible object.
        # In this task, top_left_posm, top_right_pos, bottom_left_pos, and bottom_right_pos respectively represent the current positions of the top left corner, top right corner, bottom left corner, and bottom right corner of the cloth.
        # The cloth is laid flat in the horizontal plane represented by the X-axis and the Z-axis.
        top_left_pos, top_right_pos, bottom_left_pos, bottom_right_pos = observation[:3], observation[15:18], observation[90:93], observation[105:108]
        # The action space is the horizontal movement space in the top left corner of the cloth. The actions in this space range between −0.15 and 0.15.
        action = [x_pos, z_pos]

        # Please note that target_pos is not required for this task, and action can be used or not.

