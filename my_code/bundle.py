import pickle
import time

import gtsam
from gtsam.symbol_shorthand import X, P
from gtsam import Pose3, StereoPoint2, GenericStereoFactor3D, Point3, KeyVector
from gtsam.utils import plot as g_plot
import numpy as np
from numpy import pi as pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

import tracks, utils, kitti, my_plot, utils, results
import gtsam_utils as g_utils

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.5g" % x))

class FactorGraphSLAM:
    def __init__(self, tracks_path, tracks_db=None):
        tracks_path_os = utils.path_to_current_os(tracks_path)
        self.tracks_path = tracks_path_os
        self.tracks_db = tracks_db if tracks_db else tracks.read(tracks_path_os)
        self.endframe = self.tracks_db.endframe # 2760
        self.bundle_len = 11 # number of frames in bundle
        self.init_cams()
        self.init_folders()
        self.init_results()
        self.keyframes_idx = list(range(0, self.endframe+1, self.bundle_len-1))
        self.joint_marginal_cov_mats = []
        self.relative_cov_mats = []
    
    def init_cams(self):
        self.k, self.ext_l0, self.ext_r0 = kitti.read_cameras() # k=(3,4) ext_l0/r0 (4,4)
        fx, skew, cx, _, fy, cy = self.k[0:2, 0:3].flatten()
        baseline = self.ext_r0[0, 3]
        self.gt_k = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -baseline)
    
    def init_folders(self):
        self.out_path = utils.path_to_linux(self.tracks_db.args.out_path)
        self.stage_3_dir = os.path.join(self.out_path, 'stage3')
        utils.clear_and_make_dir(self.stage_3_dir)
        self.single_bundle_plot_dir = os.path.join(self.stage_3_dir, 'single_bundle')
        utils.clear_and_make_dir(self.single_bundle_plot_dir)

    def init_results(self):
        self.init_errors, self.final_errors = [], []
        self.stats = ["**Stage3**", "tracks_db args:", str(self.tracks_db.args), self.tracks_path,
                      'optimizer.error() before optimizer.optimize() and after', f'bundle_len={self.bundle_len}']
        self.save = True
        self.single_bundle_plots = True

    def main(self):
        Pose3_keyframes = gtsam.Values()
        Pose3_keyframes.insert(X(0), Pose3())
        start_time = time.time()
        for l0_idx in self.keyframes_idx[:-1]:
            endframe = l0_idx+self.bundle_len-1 # 10, in [0-10] inclusive
            lk_to_l0_Pose3 = self.single_bundle(l0_idx=l0_idx)
            Pose3_keyframes.insert(X(endframe), lk_to_l0_Pose3)
        self.output_results(Pose3_keyframes, start_time)
    
    def single_bundle(self, l0_idx): # bundelon
        graph, initialEstimate = self.build_graph(l0_idx)

        # optimize
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        self.init_errors.append(optimizer.error())
        values = optimizer.optimize()
        
        # extract results
        endframe = l0_idx + self.bundle_len - 1 # 10
        marginals = gtsam.Marginals(graph, values)
        pose_l0_idx_to_l0 = values.atPose3(X(l0_idx))
        pose_l_endframe_to_l0 = values.atPose3(X(endframe))
        pose_l_endframe_to_l0_idx = pose_l0_idx_to_l0.between(pose_l_endframe_to_l0)
        keys = KeyVector([X(l0_idx), X(endframe)])
        joint_covariance_l0idx_l_endframe = marginals.jointMarginalCovariance(keys).fullMatrix() # (12,12) array
        relative_info_l_endframe_cond_on_l0_idx = marginals.jointMarginalInformation(keys).at( X(endframe), X(endframe) )
        relative_cov_l_endframe_cond_on_l0_idx = g_utils.relative_cov_li_cond_on_l0(marginals, li_idx=endframe, l0_idx=l0_idx)
        self.joint_marginal_cov_mats.append(joint_covariance_l0idx_l_endframe)
        self.relative_cov_mats.append(relative_cov_l_endframe_cond_on_l0_idx)
        pose_x_endframe = values.atPose3( X(endframe) )
        self.final_errors.append(optimizer.error())

        # add stats and plots
        msg = f'bundle frames [{l0_idx}-{endframe}]: error before: {self.init_errors[-1]:.1f}, after: {self.final_errors[-1]:.1f}'
        self.stats.append(msg)
        print(msg)
        
        if self.single_bundle_plots:
            g_utils.single_bundle_plots(values, self.single_bundle_plot_dir, endframe, l0_idx)
            g_utils.my_cond_plot_trajectory(l0_idx+2, values, marginals, l0_idx, endframe, plot_dir=self.single_bundle_plot_dir)
            a=2

        return pose_x_endframe
    
    def build_graph(self, l0_idx):
        endframe = l0_idx + self.bundle_len - 1 # 10
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()
        pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1*pi/180, 1*pi/180, 1*pi/180, 0.3, 0.3, 0.3])) # (1 deg, input must be radian), 0.3 meter
        meas_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

        # add Initial Estimates for Camera
        l0_to_li_s = self.tracks_db.ext_l1s[l0_idx:endframe+1]
        li_to_l0_s = [utils.inv_extrinsics(ext_mat) for ext_mat in l0_to_li_s]
        for i in range(l0_idx , endframe+1):
            initialEstimate.insert(X(i), Pose3(li_to_l0_s[i-l0_idx]))

        # add Prior Factor for Pose
        priorFactor = gtsam.PriorFactorPose3(X(l0_idx), Pose3(li_to_l0_s[0]), pose_noise_model)
        graph.add(priorFactor)
        # graph.add( gtsam.NonlinearEqualityPose3(X(l0_idx), Pose3()) )

        # Add factors for all measurements
        for cam_idx in range(l0_idx, endframe+1):
            # get tracks
            idx_tracks = self.tracks_db.get_tracks(cam_id=cam_idx)
            for track in idx_tracks: # add factor for each track
                # filter irrelevant tracks 
                if cam_idx==l0_idx and track.next is None:
                    continue
                if track.orig_cam_id == endframe:
                    continue
                
                # add Factor for measurement
                stereo_point = StereoPoint2(track.left_x, track.right_x, track.left_y)
                stereoFactor = GenericStereoFactor3D(stereo_point, meas_noise_model, X(cam_idx), P(track.id), self.gt_k)
                graph.add(stereoFactor)
                
                # if new point in bundle, add initial estimate
                if cam_idx <= max(track.orig_cam_id, l0_idx):
                    pc_li = track.pc
                    initialEstimate.insert( P(track.id), Point3(pc_li))
        return graph, initialEstimate
    
    def output_results(self, Pose3_keyframes, start_time):
        # output important plots and stats
        ext_l0_to_l1_s = g_utils.get_world_to_cam_ext_from_values(Pose3_keyframes)
        rots_total_error, trans_total_error = results.output_results(self.out_path, ext_l0_to_l1_s, self.keyframes_idx, "stage3", start_time, relative=False, plot=False, save=self.save)

        # rename stage3 folder
        new_stage3_dir = self.stage_3_dir + f'_{trans_total_error:.1f}_{rots_total_error:.1f}'
        new_stage3_dir = utils.get_avail_path(new_stage3_dir) # TODO remove
        os.rename(self.stage_3_dir, new_stage3_dir)
        self.stage_3_dir = new_stage3_dir
        
        # write stats
        with open (os.path.join(self.stage_3_dir,'stats_stage3.txt'), 'w') as f:
            f.writelines('\n'.join(self.stats))
        
        # graph before_after optimization errors
        my_plot.plt_bundle_errors(self.init_errors, self.final_errors, self.stage_3_dir, idx=self.keyframes_idx[1:], plot=False, save=self.save)

        # serialzie Pose3 and marginals
        g_utils.serialize_Pose3_marginals(self.stage_3_dir, Pose3_keyframes, self.joint_marginal_cov_mats, self.relative_cov_mats, self.keyframes_idx)


if __name__=="__main__":
    tracks_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/06-13-16-50_mine_global_50/stage2_1.3_1.7/stage2_tracks_50_filtered.pkl'
    # tracks_path =  r'/mnt/c/users/godin/Documents/VAN_ex/out/06-13-12-25_mine_global_500/stage2_57.2_18.6/stage2_tracks_500_filtered.pkl'
    # tracks_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/06-10-15-12_mine_global_2760/stage2_622.2_114.9/stage2_tracks_2760_filtered.pkl'
    # tracks_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/06-13-14-30_mine_global_200/stage2_19.0_8.5/stage2_tracks_200_filtered.pkl'
    
    ba = FactorGraphSLAM(tracks_path=tracks_path)
    ba.main()
    print('bundle end')