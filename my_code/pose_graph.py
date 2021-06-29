import numpy as np
from numpy import pi as pi
import gtsam
from gtsam.symbol_shorthand import X, P
from gtsam import Pose3, StereoPoint2, GenericStereoFactor3D, Point3, KeyVector
from gtsam.utils import plot as g_plot
import matplotlib.pyplot as plt

import os

import utils, my_plot
import gtsam_utils as g_utils

class PoseGraph:
    def __init__(self, pkl_path):
        pkl_os_path = utils.path_to_current_os(pkl_path)
        self.stage3_pkl_dir, _, _ = utils.dir_name_ext(pkl_os_path)
        self.pkl_path = pkl_os_path
        self.Pose3_values, self.joint_marginal_cov_mats, self.cond_cov_mats,\
             self.keyframes_idx = g_utils.unserialize_Pose3_marginals(pkl_os_path) # [0,10,20,...,2760], 277
        self.cond_cov_cumsum = g_utils.cumsum_mats(self.cond_cov_mats) # 277. a[i]= Sigma_i|0
        self.cond_cov_cumsum.insert(0, np.zeros((6,6)))
        self.init_errors, self.final_errors = [], []
        self.stats = [pkl_os_path]
        self.stage4_plot_dir = os.path.join( os.path.dirname(self.stage3_pkl_dir), 'stage4' )
        utils.make_dir_if_needed(self.stage4_plot_dir)
        self.error_plot_dir = os.path.join(self.stage4_plot_dir, 'errors')
        utils.clear_and_make_dir(self.error_plot_dir)

    def ck_cond_ci_cov(self, lk_idx, li_idx): # e.g. 40,20
        lk_kf_num = self.keyframes_idx.index(lk_idx) # 4
        li_kf_num = self.keyframes_idx.index(li_idx) # 2
        ck_cond_ci_cov = self.cond_cov_cumsum[lk_kf_num] - self.cond_cov_cumsum[li_kf_num] # (6,6)
        return ck_cond_ci_cov
    
    def comp_mahal_dist(self, n_kf_idx, save=False): # e.g. 20 - 
        mahal_dists = [] 
        n_kf_num = self.keyframes_idx.index(n_kf_idx) # e.g. 20 = 2, since [0,10,20,...]
        pose_cn = self.Pose3_values.atPose3( X(n_kf_idx) )
        cov_cn_cond_ci = self.cond_cov_cumsum[n_kf_num]
        for kf_num, kf_idx in enumerate(self.keyframes_idx[0:n_kf_num]):
            pose_ci = self.Pose3_values.atPose3( X(kf_idx) )
            cn_to_ci = pose_ci.between(pose_cn) # from cn to ci
            t2v = utils.t2v(cn_to_ci) # (6,)

            cov_cn_cond_ci -= self.cond_cov_cumsum[kf_num] # (6,6)

            mahal_dist = t2v.T @ cov_cn_cond_ci @ t2v
            mahal_dists.append(mahal_dist)
        
        if save:
            my_plot.plotly_scatter(x=self.keyframes_idx[0:n_kf_num], y=mahal_dists, mode='lines+markers',
                               name=f"0_{n_kf_idx}_mahal_dists", plot_dir=self.error_plot_dir, plot=False)
        return mahal_dists        
    
    def make_between_factor(self,li_idx, lk_idx): # keyframe idx, i.e. 20,40
        pose_ci = self.Pose3_values.atPose3( X(li_idx) )
        pose_ck = self.Pose3_values.atPose3( X(lk_idx) )
        ck_to_ci_pose = pose_ci.between(pose_ck)
        lk_cond_li_cov = self.ck_cond_ci_cov(lk_idx, li_idx)
        noise_model = gtsam.noiseModel.Gaussian.Covariance(lk_cond_li_cov)
        factor = gtsam.BetweenFactorPose3( X(li_idx), X(lk_idx) , ck_to_ci_pose, noise_model)
        return factor

    
    def optimize(self):
        endframe = self.keyframes_idx[-1]
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()
        
        # add initial estimate
        for keyframe_idx  in self.keyframes_idx:
            lk_to_l0_pose = self.Pose3_values.atPose3( X(keyframe_idx) )
            initialEstimate.insert( X(keyframe_idx), lk_to_l0_pose )
        
        # add prior factor

        pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1*pi/180, 1*pi/180, 1*pi/180, 0.3, 0.3, 0.3])) # (1 deg, input must be radian), 0.3 meter
        init_pose = self.Pose3_values.atPose3( X(0) )
        priorFactor = gtsam.PriorFactorPose3( X(0), init_pose, pose_noise_model)
        graph.add(priorFactor)
        
        # add between factors
        for l0_kf_num in range( len(self.keyframes_idx)-1 ): # [0,1,2,...,275]
            l0_idx, l1_idx = self.keyframes_idx[l0_kf_num], self.keyframes_idx[l0_kf_num+1] # [0, 10]
            pose_c0 = self.Pose3_values.atPose3( X(l0_idx) )
            pose_c1 = self.Pose3_values.atPose3( X(l1_idx) )
            c1_to_c0_pose = pose_c0.between(pose_c1)
            l1_cond_l0_cov = self.cond_cov_mats[l0_kf_num] # ndarray (6,6)
            noise_model = gtsam.noiseModel.Gaussian.Covariance(l1_cond_l0_cov)
            factor = gtsam.BetweenFactorPose3( X(l0_idx), X(l1_idx) , c1_to_c0_pose, noise_model)
            factor_error = factor.error(initialEstimate)
            graph.add(factor)
            # compute errors
            mahal_dists = self.comp_mahal_dist(l1_idx, save=True)
            factors = []
            for l_prev_idx in self.keyframes_idx[:(l0_kf_num+1)]:
                factors.append(self.make_between_factor(l_prev_idx, l1_idx))
            errors = [fact.error(initialEstimate) for fact in factors]
            my_plot.plotly_scatter(x=self.keyframes_idx[0:(l0_kf_num+1)], y=errors, mode='lines+markers',
                               name=f"0_{l1_idx}_factor_errors", plot_dir=self.error_plot_dir, plot=False)

            if l0_kf_num and l0_kf_num % 20 == 0:
                print(f'finished {l0_kf_num}')
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
    # pg.comp_mahal_dist(n_kf_idx=1320, save=True)
    pg.optimize()
    

