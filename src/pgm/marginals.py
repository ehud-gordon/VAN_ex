""" Class for storing and computing conditional covariances between poses. """

import numpy as np
from collections import defaultdict
from enum import Enum, auto
import pgm.utils as g_utils
WORLD = 0

class Status(Enum):
    GET_COND_COV = auto()
    GET_PATH_COND_COV = auto()
    SET_COND_COV = auto()

class Marginals:
    def __init__(self) -> None:
        self.marginals = defaultdict(dict)
        self.marginals[WORLD][WORLD] = np.zeros((6,6))
        self.status = ""
        self.source_idx = 0
    
    def get_cov_of(self, source_idx):
        """ usage: marginals.get_cov_of(source).conditional_on(target) """
        self.status = Status.GET_COND_COV
        self.source_idx = source_idx
        return self

    def get_path_cov_of(self, source_idx):
        """ usage: marginals.get_path_cov_of(source).conditional_on(target).along_path(path) """
        self.status = Status.GET_PATH_COND_COV
        self.source_idx = source_idx
        return self
    
    def set_cov_of(self, source_idx):
        """ usage: marginals.set_cov_of(source).conditional_on(target).with_(cov_matrix) """
        self.status = Status.SET_COND_COV
        self.source_idx = source_idx
        return self
    
    def conditional_on(self, target_idx):
        if self.status == Status.GET_COND_COV:
            return self.marginals[self.source_idx][target_idx]
        elif self.status == Status.GET_PATH_COND_COV:
            self.target_idx = target_idx
            return self
        elif self.status == Status.SET_COND_COV:
            self.target_idx = target_idx
            return self
    
    def with_(self, cov_matrix):
        """ usage: marginals.set_cov_of(source).conditional_on(target).with_(cov_matrix)
        sets covariance of source contidional on target

        :param cov_matrix: ndarry, covariance matrix
        """
        # update covariance of source conditional on target
        self.marginals[self.source_idx][self.target_idx] = cov_matrix
        # update covariance of source conditional on world
        cov_target_conditional_on_world = self.marginals[self.target_idx][WORLD]
        self.marginals[self.source_idx][WORLD] = cov_target_conditional_on_world + cov_matrix

    def along_path(self, path_frames):
        """ usage: marginals.get_path_cov_of(source).conditional_on(target).along_path(path_frames).
         e.g. if path_frames=[0,10,20,30], returns
         cov_of_30_conditional_on_0 = 10_cond_on_0 + 20_cond_on_10 + 30_cond_on_20

         :param path_frames: [target, ... , source]
         :return: covariance of source conditional on target
         """
        assert path_frames[0] == self.target_idx and path_frames[-1] == self.source_idx
        cov_target_conditional_on_source = np.zeros((6,6))
        for i, j in zip(path_frames[:-1], path_frames[1:]):
            if j >= i:
                cov_j_conditional_on_i = self.marginals[j][i]
            if j < i:
                cov_j_conditional_on_i = self.marginals[i][j] # this works
            cov_target_conditional_on_source += cov_j_conditional_on_i
        return cov_target_conditional_on_source

    def get_path_covs(self, path_frames):
        """ gets path [target, frame1..., frame_n] and returns (estimation of) the covariances of frame_i conditional on target.
        The estimation is done by summing the conditional covariances between consecutive frames along the path.
        e.g. if path is [10,20,30] then reutrns [Sigma_{20|10}, Sigma_{30|10}]
        with Sigma_{30|10} = Sigma_{20|10} + Sigma_{30|20}

        :param path_frames: list of n+1 frames [target, frame1, ...., frame_n]
        :return: list of n conditional covariances matrices:
                 [Sigma_{frame1|target}, ..., Sigma_{frame_n|target}]
        """
        covs_conditional_on_target = [np.zeros((6,6))]
        for i,j in zip(path_frames[:-1], path_frames[1:]):
            cov_j_conditional_on_i = self.marginals[j][i]
            cov_i_conditiona_on_target = covs_conditional_on_target[-1]
            cov_j_conditional_on_target = cov_j_conditional_on_i + cov_i_conditiona_on_target
            covs_conditional_on_target.append(cov_j_conditional_on_target)

        return covs_conditional_on_target[1:]

    def update_with_Marginals(self, frames_idx, marginals):
        """ updates conditional covariances with bundle marginals.

        :param frames_idx: list of n+1 frames [startframe, frame1,..., frame_n]
        :param marginals: gtsam.Marginals object
        """
        startframe = frames_idx[0]
        for i in frames_idx[1:]:
            # update cov_i_conditional_on_startframe
            self.marginals[i][startframe] = g_utils.extract_conditional_covariance(marginals, startframe, i)

