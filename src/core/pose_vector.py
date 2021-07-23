import numpy as np
import gtsam
from gtsam import Pose3

import utils.geometry


class PoseVector:
    def __init__(self, list_of_poses):
        assert len(list_of_poses) > 0
        self.poses = list_of_poses
        if gtsam.__name__ not in str(type(list_of_poses[0])):
            self.poses = [Pose3(pose) for pose in self.poses]

    def __len__(self):
        return len(self.poses)

    def __repr__(self):
        return str(self.poses)

    def __getstate__(self):
        np_poses = self.as_np()
        return np_poses
    
    def __setstate__(self, np_poses):
        self.poses = [Pose3(mat) for mat in np_poses]
    
    def __getitem__(self, item):
        return self.poses[item]

    def get_translations(self):
        return np.array([pose.translation() for pose in self.poses]).T

    def get_dws(self):
        return self.get_translations()

    def get_rotations(self):
        return [pose.rotation().matrix() for pose in self.poses]

    def get_rots_trans(self):
        trans_vecs = self.get_translations()
        rotations = self.get_rotations()
        return trans_vecs, rotations

    def compare(self, pose_vector):
        """ return the size differences between each two matching poses

        :param pose_vector: PoseVector of size n
        :return:
            rot_diffs - (n,) of n degrees approximating the size of rotation between two matching rotation matrices
            trans_diffs - (n,3) L2 distnace between two matching translation vectors
        """
        assert len(self) == len(pose_vector)
        rot_diffs, trans_diffs = [], []
        for pose1, pose2 in zip(self.poses, pose_vector):
            rot1 = pose1.rotation().matrix()
            rot2 = pose2.rotation().matrix()
            rot_diff_in_deg = utils.geometry.rotation_matrices_diff(rot1, rot2)
            rot_diffs.append(rot_diff_in_deg)
            trans_diffs.append(np.abs(pose1.translation() - pose2.translation()))
        return np.array(rot_diffs), np.array(trans_diffs)

    def get_relative(self):
        """ return PoseVector with poses_j_to_i between indices i,j in this PoseVector """
        poses_j_to_i = [Pose3()]
        for pose_i_to_world, pose_j_to_world in zip(self.poses[:-1], self.poses[1:]):
            pose_j_to_i = pose_i_to_world.between(pose_j_to_world)
            poses_j_to_i.append(pose_j_to_i)
        return PoseVector(poses_j_to_i)

    def as_np(self):
        return [pose.matrix() for pose in self.poses]

