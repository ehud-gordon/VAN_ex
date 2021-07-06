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

from numpy.random import f

import tracks, utils, kitti, my_plot, utils, results
import gtsam_utils as g_utils

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.5g" % x))

class FactorGraphSLAM:
    def __init__(self, tracks_db, out_path):
        self.tracks_db = tracks_db
        frames_idx = utils.sorted_nums_form_dict_keys(tracks_db.td)
        self.endframe = frames_idx[-1] # 2760
        self.bundle_len = 11 # number of frames in bundle
        self.out_path = out_path
        self.init_cams()
        self.init_results()
        self.init_folders()
        self.keyframes_idx = []
        for i in range(0, len(frames_idx)+1, self.bundle_len-1):
            i_kf = frames_idx[i]
            self.keyframes_idx.append(i_kf)
        self.cov_lj_cond_li_keyframes = []
    
    def init_cams(self):
        self.k, self.ext_id, self.ext_l_to_r = kitti.read_cameras()
        self.gt_k = g_utils.get_gt_k(self.k, self.ext_l_to_r)

    def init_results(self):
        self.errors_before, self.errors_after = [], []
        self.stats = ["**Stage3**", 'errors are: optimizer.error() before optimizer.optimize(), and after', f'bundle_len={self.bundle_len}']
        self.save = True
        self.single_bundle_plots = False

    def init_folders(self):
        self.stage_3_dir = os.path.join(self.out_path, 'stage3')
        utils.clear_and_make_dir(self.stage_3_dir)
        if self.single_bundle_plots:
            self.single_bundle_plot_dir = os.path.join(self.stage_3_dir, 'single_bundle')
            utils.clear_and_make_dir(self.single_bundle_plot_dir)
    


    def main(self):
        Pose3_li_to_l0_keyframes = gtsam.Values()
        Pose3_li_to_l0_keyframes.insert(X(0), Pose3())
        start_time = time.time()
        for startframe in self.keyframes_idx[:-1]: 
            # perform single bundelon
            endframe = startframe + self.bundle_len - 1 # 10, in [0-10] inclusive
            frames_idx = list(range(startframe, startframe+self.bundle_len))
            ext_l0_to_li_s = self.tracks_db.ext_l0_to_li_s[startframe:endframe+1]
            ext_li_to_l0_s = [utils.inv_extrinsics(l0_to_li) for l0_to_li in ext_l0_to_li_s]

            values, error_before, error_after, marginals = single_bundle(frames_idx, ext_li_to_l0_s, self.tracks_db, self.gt_k)
            # output bundelon results
            cov_lend_cond_on_lstart = g_utils.cov_ln_cond_on_li(marginals, endframe, startframe)
            msg = f'bundle frames [{startframe}-{endframe}]: error before: {error_before:.1f}, after: {error_after:.1f}'
            self.stats.append(msg)
            print(msg)
            self.cov_lj_cond_li_keyframes.append(cov_lend_cond_on_lstart)
            Pose3_lend_to_l0 = values.atPose3( X(endframe) )
            self.errors_before.append(error_before)
            self.errors_after.append(error_after)
            Pose3_li_to_l0_keyframes.insert( X(endframe), Pose3_lend_to_l0 )
            if self.single_bundle_plots:
                g_utils.single_bundle_plots(values, self.single_bundle_plot_dir, startframe, endframe)
                g_utils.my_cond_plot_trajectory(startframe+2, values, marginals, startframe, endframe, plot_dir=self.single_bundle_plot_dir)

        self.output_results(Pose3_li_to_l0_keyframes, start_time)
    
    
    def output_results(self, Pose3_li_to_l0_keyframes, start_time):
        # output important plots and stats
        ext_li_to_l0_s = g_utils.ext_ci_to_c0_s_from_values(Pose3_li_to_l0_keyframes)
        rots_total_error, trans_total_error = results.output_results(self.out_path, ext_li_to_l0_s,
                            self.keyframes_idx, "stage3", start_time, relative=False, plot=False, save=self.save)

        # rename stage3 folder
        new_stage3_dir = self.stage_3_dir + f'_{trans_total_error:.1f}_{rots_total_error:.1f}'
        new_stage3_dir = utils.get_avail_path(new_stage3_dir) # TODO remove
        os.rename(self.stage_3_dir, new_stage3_dir)
        self.stage_3_dir = new_stage3_dir
        
        # write stats
        with open (os.path.join(self.stage_3_dir,'stats_stage3.txt'), 'w') as f:
            f.writelines('\n'.join(self.stats))
        
        # graph before_after optimization errors
        my_plot.plt_bundle_errors(self.errors_before, self.errors_after, self.keyframes_idx[1:], "stage3", self.stage_3_dir, plot=False)

        # serialzie Pose3 and marginals
        g_utils.serialize_bundle(self.stage_3_dir, Pose3_li_to_l0_keyframes, self.cov_lj_cond_li_keyframes, self.keyframes_idx, title="stage3")

def build_bundle_graph(frames_idx, ext_li_to_l0_s, tracks_db, gt_k):
    startframe = frames_idx[0]
    endframe = frames_idx[-1]        
    graph = gtsam.NonlinearFactorGraph()
    initialEstimate = gtsam.Values()
    pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1*pi/180, 1*pi/180, 1*pi/180, 0.3, 0.3, 0.3])) # (1 deg, input must be radian), 0.3 meter
    meas_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

    # add initial estimates for cameras
    for i_frame, ext_li_to_l0 in zip(frames_idx, ext_li_to_l0_s):
        initialEstimate.insert( X(i_frame), Pose3(ext_li_to_l0) )

    # add Prior Factor for start pose
    priorFactor = gtsam.PriorFactorPose3( X(startframe), Pose3(ext_li_to_l0_s[0]), pose_noise_model )
    graph.add(priorFactor)
    # graph.add( gtsam.NonlinearEqualityPose3(X(l0_idx), Pose3()) )

    # Add factors for all measurements
    for i_kf in frames_idx:
        # get tracks
        idx_tracks = tracks_db.get_tracks(i_kf)
        for track in idx_tracks: # add factor for each track
            # filter irrelevant tracks 
            if i_kf==startframe and track.next is None:
                continue
            if track.orig_cam_id == endframe:
                continue
            
            # add Factor for measurement
            stereo_point = StereoPoint2(track.left_x, track.right_x, track.left_y)
            stereoFactor = GenericStereoFactor3D(stereo_point, meas_noise_model, X(i_kf), P(track.id), gt_k)
            graph.add(stereoFactor)
            
            # if new point in bundle, add initial estimate
            # TODO could be ==
            if i_kf <= max(track.orig_cam_id, startframe):
                pc_li = track.pc
                initialEstimate.insert( P(track.id), Point3(pc_li))
    return graph, initialEstimate

def single_bundle(frames_idx, ext_li_to_l0_s, tracks_db, gt_k): # bundelon
        graph, initialEstimate = build_bundle_graph(frames_idx, ext_li_to_l0_s, tracks_db, gt_k)
        # optimize
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        error_before = optimizer.error()
        values = optimizer.optimize()
        error_after = optimizer.error()
        marginals = gtsam.Marginals(graph, values)
        return values, error_before, error_after, marginals

if __name__=="__main__":
    # tracks_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/06-10-15-12_global_2760/stage2_56.0_114.9/stage2_tracks_2760_filtered.pkl'
    # tracks_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/06-13-16-50_global_50/stage2_1.3_1.7/stage2_tracks_50_filtered.pkl'
    # tracks_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-02-01-24_global_2760/stage2_55.1_114.0/stage2_tracks_2760_filtered.pkl'
    tracks_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-04-17-44_global_2760/stage2_58.0_119.7/stage2_tracks_2760_filtered.pkl'
    # unfilt_tracks_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-04-17-44_global_2760/stage2_58.0_119.7/stage2_tracks_2760.pkl'
    tracks_path_os = utils.path_to_current_os(tracks_path)
    tracks_dir, _, _ = utils.dir_name_ext(tracks_path_os)
    out_path = os.path.dirname(tracks_dir)
    tracks_db = tracks.read(tracks_path_os)
    ba = FactorGraphSLAM(tracks_db, out_path)
    ba.main()
    print('bundle end')

def find_where_in_grahp(graph, num=8070450532252330524, ppoint="p4401692"):
    for i in range(graph.size()):
        fact_i = graph.at(i)
        if num in fact_i.keys():
            print(fact_i)