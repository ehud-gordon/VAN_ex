from collections import defaultdict
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
    def __init__(self, ext_li_to_lj_s, tracks_db, out_path):
        self.tracks_db = tracks_db
        self.frames_idx = tracks_db.frames_idx # [0,...10,11,...20]
        self.endframe = self.frames_idx[-1] # 2760
        self.bundle_len = 11 # number of frames in bundle
        self.out_path = out_path
        self.ext_li_to_lj_s = ext_li_to_lj_s # ext[j] = lj_to_li
        assert len(self.frames_idx) == len(ext_li_to_lj_s)
        assert np.allclose(ext_li_to_lj_s[0], np.diag([1,1,1,1]))
        self.keyframes_idx = self.frames_idx[::(self.bundle_len-1)] #[0,10,...,2760]
        self.init_cams()
        self.init_results()
        self.init_folders()
    
    def init_cams(self):
        self.k, self.ext_id, self.ext_l_to_r = kitti.read_cameras()
        self.gt_k = g_utils.get_gt_k(self.k, self.ext_l_to_r)

    def init_results(self):
        self.errors_before, self.errors_after = [], []
        self.stats = ["**Stage3**", 'errors are: optimizer.error() before optimizer.optimize(), and after', f'bundle_len={self.bundle_len}']
        self.save = True
        self.single_bundle_plots = False

    def init_folders(self):
        self.stage3_dir = os.path.join(self.out_path, 'stage3')
        utils.clear_and_make_dir(self.stage3_dir)
        if self.single_bundle_plots:
            self.single_bundle_plot_dir = os.path.join(self.stage3_dir, 'single_bundle')
            utils.clear_and_make_dir(self.single_bundle_plot_dir)
    
    def main(self):
        Pose3_lj_to_li_keyframes = []
        cov_lj_cond_li_dict = defaultdict(dict)
        Pose3_lj_to_li_keyframes.append(Pose3())
        start_time = time.time()
        num_keyframes = len(self.keyframes_idx) # 277
        for j in range(0,num_keyframes-1): # [0,..,275]
        # for i in range(0, len(self.frames_idx)-1, self.bundle_len-1): # [0, 10, ..., 2750]
            # perform single bundelon
            i = self.keyframes_idx[j] # [0, 10, ..., 2750]
            bundelon_frames_idx = self.frames_idx[i:(i+self.bundle_len)] #[20,...,30]
            startframe = bundelon_frames_idx[0]; endframe = bundelon_frames_idx[-1]
            bundelon_ext_li_to_lj_s = self.ext_li_to_lj_s[ (i+1):(i+self.bundle_len) ]  # [21-to_20, ,...,30_to_29]
            bundelon_ext_li_to_lj_s.insert(0, self.ext_id) # # [id, 21-to_20, ,...,30_to_29]
            bundelon_ext_li_to_l0_s = utils.concat_and_inv_ci_to_cj_s(bundelon_ext_li_to_lj_s) # [l20_to_l20, l21_to_l20,...,l30_to_l20]
            values, error_before, error_after, marginals = do_single_bundle(bundelon_frames_idx, bundelon_ext_li_to_l0_s, self.tracks_db, self.gt_k)
            # output bundelon results
            cov_lend_cond_on_lstart = g_utils.extract_cov_ln_cond_li_from_marginals(marginals, startframe, endframe)
            cov_lstart_cond_on_lend = g_utils.extract_cov_ln_cond_li_from_marginals(marginals, endframe, startframe)
            cov_lj_cond_li_dict[j+1][j] = cov_lend_cond_on_lstart
            cov_lj_cond_li_dict[j][j+1] = cov_lstart_cond_on_lend
            msg = f'bundle frames [{startframe}-{endframe}]: error before: {error_before:.1f}, after: {error_after:.1f}'
            self.stats.append(msg)
            print(msg)
            Pose3_lstart = values.atPose3( X(startframe) )
            Pose3_lend = values.atPose3( X(endframe) )
            Pose3_lend_to_lstart = Pose3_lstart.between(Pose3_lend)
            self.errors_before.append(error_before)
            self.errors_after.append(error_after)
            Pose3_lj_to_li_keyframes.append(Pose3_lend_to_lstart)
            if self.single_bundle_plots:
                g_utils.single_bundle_plots(values, self.single_bundle_plot_dir, startframe, endframe, marginals)

        self.output_results(Pose3_lj_to_li_keyframes, cov_lj_cond_li_dict, start_time)

    def output_results(self, Pose3_lj_to_li_keyframes, cov_lj_cond_li_dict, start_time):
        # output important plots and stats
        ext_lj_to_li_keyframes = [pose.matrix() for pose in Pose3_lj_to_li_keyframes]
        ext_li_to_l0_keyframes = utils.concat_cj_to_ci_s(ext_lj_to_li_keyframes)
        rots_total_error, trans_total_error = results.output_results(self.out_path, ext_li_to_l0_keyframes,
                            self.keyframes_idx, "stage_3", start_time, plot=False, save=self.save)

        # rename stage3 folder
        new_stage3_dir = self.stage3_dir + f'_{trans_total_error:.1f}_{rots_total_error:.1f}'
        new_stage3_dir = utils.get_avail_path(new_stage3_dir)
        os.rename(self.stage3_dir, new_stage3_dir)
        self.stage3_dir = new_stage3_dir
        
        # write stats
        with open (os.path.join(self.stage3_dir,'stats_stage3.txt'), 'w') as f:
            f.writelines('\n'.join(self.stats))

        # plot determinant of covariance matrices
        my_plot.plotly_cov_dets(cov_lj_cond_li_dict, self.keyframes_idx, title="stage3", plot_dir=self.stage3_dir, plot=False, save=self.save)
        
        # plot before and after optimization errors
        tmp_d = {'errors_before':self.errors_before, 'errors_after':self.errors_after}
        my_plot.plotly_scatters(tmp_d, x=self.keyframes_idx[1:], title="errors_before_after_stage3", plot_dir=self.stage3_dir, plot=False, save=self.save,
                               yaxis="error")
        # my_plot.plt_bundle_errors(self.errors_before, self.errors_after, self.keyframes_idx[1:], "stage3", self.stage3_dir, plot=False)

        # serialzie Pose3 and marginals
        g_utils.serialize_bundle(self.stage3_dir, ext_lj_to_li_keyframes, cov_lj_cond_li_dict, self.keyframes_idx, title="stage3")

def build_bundle_graph(frames_idx, ext_li_to_lstart_s, tracks_db, gt_k):
    assert len(frames_idx) == len(ext_li_to_lstart_s) # [l20_to_l20, l21_to_l20,...,l30_to_l20]
    assert np.allclose( ext_li_to_lstart_s[0], np.diag([1,1,1,1]) )
    startframe = frames_idx[0]
    endframe = frames_idx[-1]        
    graph = gtsam.NonlinearFactorGraph()
    initialEstimate = gtsam.Values()
    pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1*pi/180, 1*pi/180, 1*pi/180, 0.3, 0.3, 0.3])) # (1 deg, input must be radian), 0.3 meter
    meas_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

    # add initial estimates for cameras
    for i_frame, ext_li_to_lstart in zip(frames_idx, ext_li_to_lstart_s):
        initialEstimate.insert( X(i_frame), Pose3(ext_li_to_lstart) )

    # add Prior Factor for start pose
    priorFactor = gtsam.PriorFactorPose3( X(startframe), Pose3(), pose_noise_model )
    graph.add(priorFactor)
    # graph.add( gtsam.NonlinearEqualityPose3(X(l0_idx), Pose3()) )

    # Add factors for all measurements
    for i, frame in enumerate(frames_idx): # [20,...,30]
        # get tracks
        idx_tracks = tracks_db.get_tracks(frame)
        for track in idx_tracks: # add factor for each track
            # filter irrelevant tracks 
            if frame==startframe and track.next is None:
                continue
            if track.orig_cam_id == endframe:
                continue
            
            # add Factor for measurement
            stereo_point = StereoPoint2(track.left_x, track.right_x, track.left_y)
            stereoFactor = GenericStereoFactor3D(stereo_point, meas_noise_model, X(frame), P(track.id), gt_k)
            graph.add(stereoFactor)
            
            # if new point in bundle, add initial estimate
            if frame == max(track.orig_cam_id, startframe):
                # convert point to startframe CS
                track_pc = track.pc # in li CS
                track_pc = ext_li_to_lstart_s[i] @ np.hstack((track_pc,1))
                track_pc = track_pc[:3]
                initialEstimate.insert( P(track.id), Point3(track_pc))
    return graph, initialEstimate

def do_single_bundle(frames_idx, ext_li_to_lstart_s, tracks_db, gt_k): # bundelon
        assert len(frames_idx) == len(ext_li_to_lstart_s)
        assert np.allclose( ext_li_to_lstart_s[0], np.diag([1,1,1,1]) )
        graph, initialEstimate = build_bundle_graph(frames_idx, ext_li_to_lstart_s, tracks_db, gt_k)
        # optimize
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        error_before = optimizer.error()
        values = optimizer.optimize()
        error_after = optimizer.error()
        marginals = gtsam.Marginals(graph, values)
        return values, error_before, error_after, marginals

if __name__=="__main__":
    # ext_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-10-14-49_0_20/stage2_0.4_0.8/ext_li_to_lj_s_stage2_20.pkl'
    # tracks_pkl_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-10-14-49_0_20/stage2_0.4_0.8/stage2_tracks_20.pkl'

    ext_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-10-13-49_0_2760/stage2_54.9_112.8/ext_li_to_lj_s_stage2_2760.pkl'
    tracks_pkl_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-10-13-49_0_2760/stage2_54.9_112.8/stage2_tracks_2760.pkl'
    
    tracks_pkl_path_os = utils.path_to_current_os(tracks_pkl_path)
    stage2_dir, _, _ = utils.dir_name_ext(tracks_pkl_path_os)
    out_path = os.path.dirname(stage2_dir)
    
    tracks_db = tracks.read(tracks_pkl_path_os)
    exts = utils.deserialize_ext_li_to_lj_s(ext_path)

    ba = FactorGraphSLAM(exts, tracks_db, out_path)
    ba.main()
    print('bundle end')



# Debugging method
def find_where_in_graph(graph, num=8070450532252330524, ppoint="p4401692"):
    for i in range(graph.size()):
        fact_i = graph.at(i)
        if num in fact_i.keys():
            print(fact_i)
