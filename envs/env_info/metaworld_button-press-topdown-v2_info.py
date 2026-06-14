class:
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        # The observation space is represented as a 6-tuple of the 3D Cartesian positions of the end-effector, a normalized measurement of how open the gripper is, the 3D position of the first object, the quaternion of the first object. 
        # In this task, first_obj_pos indicates the current position of the button and first_obj_quat indicates the quaternion of the button. 
        observation = np.concatenate(pos_eef[:3], gripper_distance_apart[0], 
                                     first_obj_pos[:3], first_obj_quat[:4], 
                                        )
    # The action space is a 2-tuple consisting of the change in 3D space of the end-effector followed by a normalized torque that the gripper fingers should apply. The actions in this space range between −1 and 1.
    action = [delta_x, delta_y, delta_z, gripper_torque]
    # The target position is the target position of the button
    target_pos = self._target_pos


