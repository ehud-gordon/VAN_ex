""" Class for storing poses (extrinsics matrices) and computing motion between them. """

import numpy as np
from gtsam import Pose3
from gtsam.symbol_shorthand import X

from collections import defaultdict
from enum import Enum, auto

from core.pose_vector import PoseVector

WORLD = 0

class Status(Enum):
    GET_POSE = auto()
    GET_PATH_POSE = auto()
    GET_DIST = auto()
    GET_TRANS = auto()
    SET_POSE = auto()

class Poses:
    def __init__(self):
        self.poses = defaultdict(dict)
        self.poses[WORLD][WORLD] = Pose3()
    
    def __getstate__(self):
        new_poses = defaultdict(dict)
        for j in self.poses.keys():
            for i, pose in self.poses[j].items():
                new_poses[j][i] = pose.matrix()
        return new_poses

    def __getitem__(self, item):
        return self.poses[item]
    
    def __setstate__(self, new_poses):
        self.poses = defaultdict(dict)
        for j in new_poses.keys():
            for i, ext_mat in new_poses[j].items():
                self.poses[j][i] = Pose3(ext_mat)
    
    def get_pose_from(self, source_idx):
        """ usage: poses.get_pose_from(source).to(target) """
        self.status = Status.GET_POSE
        self.source_idx = source_idx
        return self

    def get_path_pose_from(self, source_idx):
        """ usage: poses.get_path_pose_from(source).to(target).along_path(path) """
        self.status = Status.GET_PATH_POSE
        self.source_idx = source_idx
        return self

    def get_distance_from(self, source_idx):
        """ usage: poses.get_distance_from(source).to(target) """
        self.status = Status.GET_DIST
        self.source_idx = source_idx
        return self

    def get_trans_vec_from(self, source_idx):
        """ usage: poses.get_trans_vec_from(source).to(target) """
        self.status = Status.GET_TRANS
        self.source_idx = source_idx
        return self

    def set_pose_from(self, source_idx):
        """ usage: poses.set_pose_from(source).to(target) """
        self.status = Status.SET_POSE
        self.source_idx = source_idx
        return self
    
    def _get_pose_from_to(self, source_idx, target_idx):
        """ private method, try to extract relative pose in various ways """
        if source_idx == target_idx:
            return Pose3()
        try:
            return self.poses[source_idx][target_idx]
        except KeyError:
            pass
        try:
            pose_from_target_to_source = self.poses[target_idx][source_idx]
            return pose_from_target_to_source.inverse()
        except KeyError:
            raise Exception

    def to(self, target_idx):
        assert target_idx <= self.source_idx
        if self.status == Status.GET_POSE:
            return self._get_pose_from_to(self.source_idx, target_idx)
        elif self.status == Status.GET_PATH_POSE:
            self.target_idx = target_idx
            return self
        elif self.status == Status.GET_TRANS:
            pose_source_to_target = self._get_pose_from_to(self.source_idx, target_idx)
            return pose_source_to_target.translation()
        elif self.status == Status.GET_DIST:
            try:
                pose_source_to_target = self.poses[self.source_idx][target_idx]
            except KeyError:
                pose_source_to_target = Pose3()
                for j in range(self.source_idx, target_idx, -1):
                    i = j-1
                    pose_j_to_i = self.poses[j][i]
                    pose_source_to_target = pose_j_to_i.compose(pose_source_to_target)
            return np.linalg.norm(pose_source_to_target.translation())
        elif self.status == Status.SET_POSE:
            self.target_idx = target_idx
            return self
    
    def with_(self, value):
        """ usage: poses.set_pose_from(source).to(target).with_(value)
        updates pose from source to target.

        :param value: either ndarray or gtasm.Pose3 objects
        """
        self.poses[self.source_idx][self.target_idx] =  Pose3(value)

    def along_path(self, path_frames):
        """ usage: poses.get_path_pose_from(source).to(target).along_path(path_frames).
        e.g. if path_frames=[0,10,20,30], returns
        pose_from_30_to_0 = 10_to_0 @ 20_to_10 @ 30_to_20

        :param path_frames: [target, ... , source]
        :return: pose from source to target
        """
        assert path_frames[0] == self.target_idx and path_frames[-1] == self.source_idx
        pose_from_source_to_target = Pose3()
        for i, j in zip(path_frames[:-1], path_frames[1:]): # usually i=10, j=20, but sometimes  j=20, i=10
            pose_from_j_to_i = self._get_pose_from_to(j, i)
            pose_from_source_to_target = pose_from_source_to_target.compose(pose_from_j_to_i)
        return pose_from_source_to_target

    def get_cumulative_path_poses(self, path_frames, as_np=False):
        """ gets path [target, frame1..., frame_n] and computes cumulative poses from frames to target, using frames along the path.
        e.g. if path is [0,10,20,30] then reutrns [0_to_0, 10_to_0, 20_to_0, 30_to_10],
        with 20_to_0 = 10_to_0 @ 20_to_10

        :param path_frames: list of n+1 frames [target, frame1, ...., frame_n]
        :param as_np: If True, return list of numpy arrays
        :return: PoseVector of len n+1 poses: [id, pose_from_frame1_to_target, ... ,pose_from_frame_n_to_target]
        """
        poses_to_target = [Pose3()]
        for i,j in zip(path_frames[:-1], path_frames[1:]):
            pose_from_j_to_i = self.poses[j][i]
            pose_from_i_to_target = poses_to_target[-1]
            pose_from_j_to_target = pose_from_i_to_target.compose(pose_from_j_to_i)
            poses_to_target.append(pose_from_j_to_target)
        if as_np:
            return [pose.matrix() for pose in poses_to_target]
        else:
            return PoseVector(poses_to_target)

    def get_path_poses(self, path_frames, as_np=False):
        """ gets path [frame1, ..., frame_n] and returns the relative poses along the path.
        e.g. if path is [0,10,20] returns [0_to_0, 10_to_0, 20_to_10].

        :param path_frames: list of n frames
        :param as_np: If True, return list of numpy arrays
        :return: list of n poses of type frame_(i)_to_frame_(i-1)
        """
        poses = [Pose3()]
        for i, j in zip(path_frames[:-1], path_frames[1:]):
            poses.append(self.poses[j][i])
        if as_np:
            return [pose.matrix() for pose in poses]
        else:
            return PoseVector(poses)

    def update_with_Values(self, values, edge_list):
        for i,j in edge_list:
            self.poses[j][i] = values.atPose3(X(i)).between(values.atPose3(X(j)))

