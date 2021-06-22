import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, P
from gtsam import Pose3, StereoPoint2, GenericStereoFactor3D, Point3, KeyVector
from gtsam.utils import plot as g_plot
import matplotlib.pyplot as plt

import os

import utils
import gtsam_utils as g_utils



class PoseGraph:
    def __init__(self, pkl_path):
        pkl_os_path = utils.path_to_current_os(pkl_path)
        self.stage3_pkl_dir, _, _ = utils.dir_name_ext(pkl_os_path)
        self.pkl_path = pkl_os_path
        self.Pose3_values, self.joint_marginal_cov_mats, self.relative_cov_mats, self.keyframes_idx = g_utils.unserialize_Pose3_marginals(pkl_os_path)
        self.init_errors, self.final_errors = [], []
        self.stats = [pkl_os_path]

    
    def optimize(self):
        endframe = self.keyframes_idx[-1]
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()
        
        # add initial estimate
        for keyframe_idx  in self.keyframes_idx:
            l_k_to_l_0_pose = self.Pose3_values.atPose3( X(keyframe_idx) )
            initialEstimate.insert( X(keyframe_idx), l_k_to_l_0_pose )
        
        # add prior factor
        pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 1.0, 1.0, 1.0]))
        init_pose = self.Pose3_values.atPose3(  X(0) )
        priorFactor = gtsam.PriorFactorPose3( X(0), init_pose, pose_noise_model)
        graph.add(priorFactor)

        for i in range( len(self.keyframes_idx)-1 ):
            l0_idx, l1_idx = self.keyframes_idx[i], self.keyframes_idx[i+1]
            pose_c0 = self.Pose3_values.atPose3( X(l0_idx) )
            pose_c1 = self.Pose3_values.atPose3( X(l1_idx) )
            l_k_to_l_k_prev_pose = pose_c0.between(pose_c1)
            relative_cov_mat = self.relative_cov_mats[i] # ndarray (6,6)
            noise_cov = gtsam.noiseModel.Gaussian.Covariance(relative_cov_mat)
            factor = gtsam.BetweenFactorPose3( X(l0_idx), X(l1_idx) , l_k_to_l_k_prev_pose, noise_cov)
            graph.add(factor)

        
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        self.init_errors.append(optimizer.error())
        values = optimizer.optimize()
        self.final_errors.append(optimizer.error())
        marginals = gtsam.Marginals(graph, values)
        msg = f'Pose graphs [{0}-{endframe}]: error before: {self.init_errors[-1]:.1f}, after: {self.final_errors[-1]:.1f}'
        self.stats.append(msg)
        print(msg)
        
        


if __name__=="__main__":
    # pkl_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/06-13-14-30_mine_global_200/stage3_14.9_5.9/Pose3_marginals_200.pkl'
    # pkl_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/06-13-16-50_mine_global_50/stage3_0.8_0.7/Pose3_marginals_50.pkl'
    pkl_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/06-10-15-12_mine_global_2760/stage3_433.1_75.9/Pose3_marginals_2760.pkl'
    pg = PoseGraph(pkl_path)
    pg.optimize()
    

