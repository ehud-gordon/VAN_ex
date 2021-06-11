import pickle

import gtsam
from gtsam.symbol_shorthand import X, P
from gtsam import Pose3, StereoPoint2, GenericStereoFactor3D, Point3
from gtsam.utils import plot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os

import tracks, utils, kitti, my_plot, utils
import gtsam_utils as g_utils

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.4g" % x))

class FactorGraphSLAM:
    def __init__(self, tracks_path):
        self.tracks_path = tracks_path
        self.tracks_db = tracks.read(tracks_path)
        self.out_path = utils.path_to_linux(self.tracks_db.args.out_path)
        self.kitti = self.tracks_db.args.kitti
        self.endframe = self.tracks_db.endframe # 2760
        self.bundle_len = 11 # number of frames in bundle
        self.k, self.ext_l0, self.ext_r0 = kitti.read_cameras() # k=(3,4) ext_l0/r0 (4,4)
        fx, skew, cx, _, fy, cy = self.k[0:2, 0:3].flatten()
        baseline = self.ext_r0[0, 3]
        self.gt_k = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -baseline)
        self.keyframes_idx = list(range(0, self.endframe+1, self.bundle_len-1))
        self.stage_3_dir = os.path.join(self.out_path, 'stage3')
        utils.make_dir(self.stage_3_dir)
        self.frames_plot_dir = os.path.join(self.stage_3_dir, 'frames')
        utils.make_dir(self.frames_plot_dir)
        self.stats = ["**Stage3**", "tracks_db args:", str(self.tracks_db.args), self.tracks_path]
        self.init_errors = []
        self.final_errors = []
        self.save = True


    def main(self):
        Pose3_keyframes = gtsam.Values()
        Pose3_keyframes.insert(X(0), Pose3())

        for l0_idx in self.keyframes_idx[:-1]:
            endframe = min(l0_idx+self.bundle_len-1, self.endframe) # 50
            lk_to_l0_mine = self.factorize(l0_idx=l0_idx)
            Pose3_keyframes.insert(X(endframe), lk_to_l0_mine)
        
        pickle_path = g_utils.serialize_Pose3_values(dir_path=self.stage_3_dir, values=Pose3_keyframes, frames_idx=self.keyframes_idx)
        #### plots ####
        # plot camera locations (regular and comparison)
        kitti_dws = kitti.read_dws(idx=self.keyframes_idx) # (3,6)
        my_dws = g_utils.get_dws_from_gtsam_values(Pose3_keyframes) # (3,6)
        my_plot.plt_2d_cams(camera_dws=my_dws, plot_dir=self.stage_3_dir, title="stage3", endframe=self.endframe, plot=False, save=self.save)
        my_plot.plt_2d_cams_compare(my_dws=my_dws, kitti_dws=kitti_dws, title="stage3", plot_dir=self.stage_3_dir, endframe=self.endframe, plot=False, save=self.save)
        my_plot.plt_3d_cams_compare( my_dws=my_dws, kitti_dws=kitti_dws, title="stage3", plot_dir=self.stage_3_dir, endframe=self.endframe, plot=False, save=self.save)
        my_plot.plotly_3d_cams_compare(my_dws=my_dws,kitti_dws=kitti_dws, plot_dir=self.stage_3_dir, title="stage3",
                                    endframe=self.endframe, save=self.save)
        
        # plot relative rot_trans diffs
        r0_to_r1_s_mine, t0_to_t1_s_mine =  g_utils.r0_to_r1_s_t0_to_t1_s_from_values(values=Pose3_keyframes)
        
        r0_to_r1_s_kitti, t0_to_t1_s_kitti = kitti.read_relative_poses_world_to_cam(self.keyframes_idx)

        rot_diffs_relative = np.array([utils.rotation_matrices_diff(r, q) for r, q in zip(r0_to_r1_s_mine, r0_to_r1_s_kitti)])
        trans_diffs_relative = np.abs(t0_to_t1_s_kitti - t0_to_t1_s_mine)
        rot_trans_stats, rots_total_error, trans_total_error = utils.rot_trans_stats(rot_diffs_relative, trans_diffs_relative, endframe=self.endframe)
        self.stats += rot_trans_stats
        my_plot.plt_diff_rot_matrices(rot_diffs_relative, self.stage_3_dir, "stage3", idx=self.keyframes_idx[1:], plot=False, save=self.save)
        my_plot.plt_diff_trans_vecs(trans_diffs_relative, plot_dir=self.stage_3_dir, title="stage3", idx=self.keyframes_idx[1:], plot=False, save=self.save)
        
        # plot 3D Poses
        plot.plot_trajectory(self.endframe, Pose3_keyframes)
        plot.set_axes_equal(self.endframe)
        plt.savefig(os.path.join(self.stage_3_dir,f'3d_cams_0_{self.endframe}.png'))
        
        # graph before_after optimization errors
        my_plot.plt_bundle_errors(self.init_errors, self.final_errors, self.stage_3_dir, idx=self.keyframes_idx[1:], plot=False, save=self.save)
        
        # write stats
        print("\n".join(self.stats))
        with open (os.path.join(self.stage_3_dir,'stats.txt'), 'w') as f:
            f.writelines('\n'.join(self.stats))
        new_stage3_dir = self.stage_3_dir + f'_{trans_total_error:.1f}_{rots_total_error:.1f}'
        os.rename(self.stage_3_dir, new_stage3_dir)
        self.stage_3_dir = new_stage3_dir

    def factorize(self, l0_idx):
        endframe = l0_idx + self.bundle_len - 1 # 10
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()
        pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 1.0, 1.0, 1.0]))
        meas_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

        # add Initial Estimates for Camera
        l0_to_li_s = self.tracks_db.ext_l1s[l0_idx:endframe+1]
        li_to_l0_s = [utils.inv_extrinsics(ext_mat) for ext_mat in l0_to_li_s]
        for i in range(l0_idx , endframe+1):
            initialEstimate.insert(X(i), Pose3(li_to_l0_s[i-l0_idx]))

        # add Prior Factor for Pose
        graph.add( gtsam.PriorFactorPose3(X(l0_idx), Pose3(li_to_l0_s[0]), pose_noise_model) )
        # graph.add( gtsam.NonlinearEqualityPose3(X(l0_idx), Pose3()) )

        # Add factors for all measurements
        for cam_idx in range(l0_idx, endframe+1):
            idx_tracks = self.tracks_db.get_tracks(cam_id=cam_idx)
            for track in idx_tracks:
                # add factor for measurement
                # [(1026), (30,87), (17,87), 1, array([-22,-3,27])]
                # if we're in endframe, and this track originated in the endframe
                if cam_idx==l0_idx and track.next is None:
                    continue
                if track.orig_cam_id == endframe:
                    continue
                orig_cam_idx = max(track.orig_cam_id, l0_idx)
                stereo_point = StereoPoint2(track.left_x, track.right_x, track.left_y)
                graph.add( GenericStereoFactor3D(stereo_point, meas_noise_model, X(cam_idx), P(track.id), self.gt_k) )
                # if new point, add initial estimate
                if orig_cam_idx == cam_idx:                    
                    pc_li = track.pc
                    initialEstimate.insert( P(track.id), Point3(pc_li))

        # optimize
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)

        init_error = optimizer.error(); self.init_errors.append(init_error)
        
        values = optimizer.optimize()
        
        final_error = optimizer.error(); self.final_errors.append(final_error)
        msg = f'bundle frames [{l0_idx}-{endframe}]: Graph error before optimization: {init_error:.1f}, after: {final_error:.1f}'
        self.stats.append(msg)
        print(msg)
        pose_x_endframe = values.atPose3(X(endframe))
        # plot 2D view cameras+points
        g_utils.plot_2d_cams_points_from_gtsam_values(values=values, plot_dir=self.frames_plot_dir, endframe=endframe, startframe=l0_idx)
        # plot 3D trajectory only cameras
        plot.plot_trajectory(l0_idx, values)
        plot.set_axes_equal(l0_idx)
        plt.savefig(os.path.join(self.frames_plot_dir, f'3d_cams_{l0_idx}_{endframe}'), bbox_inches='tight', pad_inches=0)

        # plot 3D trajectory cameras+points
        plot.plot_trajectory(l0_idx+1, values)
        plot.plot_3d_points(l0_idx+1, values, linespec='r*')
        plot.set_axes_equal(l0_idx+1)
        plt.savefig(os.path.join(self.frames_plot_dir, f'3d_cams_points_{l0_idx}_{endframe}'), bbox_inches='tight', pad_inches=0)
        plt.close('all')
        return pose_x_endframe


if __name__=="__main__":
    # tracks_path = r'C:\Users\godin\Documents\VAN_ex\out\06-10-15-07_mine_global_50\stage2_1.2_1.7\stage2_tracks_50_filtered.pkl'
    tracks_path = r'C:\Users\godin\Documents\VAN_ex\out\06-10-15-09_mine_global_200\stage2_19.1_8.4\stage2_tracks_200_filtered.pkl'
    # tracks_path = r'C :\Users\godin\Documents\VAN_ex\out\06-10-15-12_mine_global_2760\stage2_622.2_114.9\stage2_tracks_2760_filtered.pkl'
    tracks_path = r'C:\Users\godin\Documents\VAN_ex\out\06-10-18-20_mine_global_2760\stage2_626.9_114.5\stage2_tracks_2760_filtered.pkl'
    l_tracks_path = utils.path_to_linux(tracks_path)
    ba = FactorGraphSLAM(tracks_path=l_tracks_path)
    ba.main()
    print('bundle end')