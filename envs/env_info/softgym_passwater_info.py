class:
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        # In this task, cup_pos_x represent the current position of the cup along X-axis. cup_length, cup_width, cup_height respectively represent the size of the cup.
        # water_height represents the height of the water in the cup. water_in_cup, water_out_of_cup respectively represent the proportion of water in and out of the cup.
        # The cup is laid in the horizontal plane represented by the X-axis and the Z-axis.
        observation = [cup_pos_x, cup_length, cup_width, cup_height, water_height, water_in_cup, water_out_of_cup]
        # The action space is the changes of the cup along the X-axis. The actions in this space range between −0.011 and 0.011. Please note that action can be used or not.
        action = [cup_pox_x_movement]
        # target_pos is the target position that the cup needs to reach


