import gtsam
from gtsam.symbol_shorthand import X, P
from gtsam import Pose3, StereoPoint2, GenericStereoFactor3D, Point3
from gtsam.utils import plot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os

import tracks
import utils
import kitti
import triang

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.4g" % x))

# TODO add prev_cam_idx and prev_match_idx to tracks
# TODO if track only in frames 9,10 don't add it to bundle of frames [10-20]
# TODO 


def get_world_to_cam_trans_vecs_from_values(values):
    cameras_keys = gtsam.utilities.allPose3s(values)
    cam_to_world_poses = [values.atPose3(k) for k in cameras_keys.keys()]
    world_to_cam_poses = [pose.inverse() for pose in cam_to_world_poses]
    world_to_cam_trans_vecs = [pose.translation() for pose in world_to_cam_poses]
    world_to_cam_trans_vecs_arr = np.array(world_to_cam_trans_vecs).T
    return world_to_cam_trans_vecs_arr

def get_dws_from_gtsam_values(values):
    cameras_keys = gtsam.utilities.allPose3s(values)
    camera_poses = [values.atPose3(k) for k in cameras_keys.keys()]
    camera_trans = [pose.translation() for pose in camera_poses]
    camera_trans = np.array(camera_trans).T
    return camera_trans

def get_points_from_gtsam_values(values):
    points = []
    for k in values.keys():
        try:
            p =values.atPoint3(k)
            points.append(p)
        except RuntimeError:
            continue
    points = np.array(points).T
    return points

def check_outlier(point):
    outlier = False
    outlier = outlier or (abs(point[0]) > 50)
    outlier = outlier or (abs(point[1]) > 50)
    outlier = outlier or (point[2] > 200) or (point[2]<1)
    return outlier

def plot_2d_cams(camera_dws, plot_dir, endframe, startframe=0):
    plt.figure()
    plt.scatter(x=camera_dws[0], y=camera_dws[2], marker=(5,2), color="red")
    plt.xlabel('x');plt.ylabel('z')
    plt.title(f"2D camera's locations for keyframes in [{startframe}-{endframe}]")
    path = os.path.join(plot_dir, f'2d_cams_{startframe}_{endframe}' + '.png')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)

def plot_2d_cams_compare(truth_dws, my_dws, plot_dir, endframe, startframe=0):
    plt.figure()
    plt.scatter(x=my_dws[0], y=my_dws[2], color="blue", label="mine")
    plt.scatter(x=truth_dws[0], y=truth_dws[2], color="red", label="truth")
    plt.xlabel('x');plt.ylabel('z')
    plt.title(f"2D camera's locations comparison for keyframes in [{startframe}-{endframe}]")
    plt.legend()
    path = os.path.join(plot_dir, f'2d_cams_comp_{startframe}_{endframe}' + '.png')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)

def plot_2d_cams_points_from_gtsam_values(values, plot_dir, endframe, startframe=0):
    dws = get_dws_from_gtsam_values(values)
    landmarks = get_points_from_gtsam_values(values)
    plt.figure()
    plt.scatter(x=dws[0], y=dws[2], color="red", marker=(5,2), label="camera")
    plt.scatter(x=landmarks[0], y=landmarks[2], color="blue", label="landmark", alpha=0.2)
    plt.xlabel('x');plt.ylabel('z')
    plt.title(f"2D cameras and landmarks for keyframes [{startframe}-{endframe}]")
    plt.legend()
    path = os.path.join(plot_dir, f'2d_cams_points_{startframe}_{endframe}' + '.png')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)

def plot_trans_vecs_diff(my_vecs, truth_vecs, idx, plot_dir, title):
    diff = np.abs(my_vecs - truth_vecs)
    fig = plt.figure()
    plt.plot(idx, diff[0], label="tx")
    plt.plot(idx, diff[1], label="ty")
    plt.plot(idx, diff[2], label="tz")
    plt.ylabel("Diff"); plt.xlabel("frames")
    plt.title (f"Difference  between my and truth translation vectors, keyframes in [{idx[0]}-{idx[-1]}]")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'trans_vecs_diff_{idx[0]}_{idx[-1]}.png'))


class FactorGraphSLAM:
    def __init__(self, tracks_path):
        self.tracks_path = tracks_path
        self.tracks_db = tracks.read(tracks_path)
        self.kitti = self.tracks_db.args.kitti
        self.endframe = self.tracks_db.endframe # 2760
        self.bundle_len = 11 # number of frames in bundle
        self.k, self.ext_l0, self.ext_r0 = kitti.read_cameras() # k=(3,4) ext_l0/r0 (4,4)
        fx, skew, cx, _, fy, cy = self.k[0:2, 0:3].flatten()
        baseline = self.ext_r0[0, 3]
        self.gt_k = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -baseline)
        self.plot_dir = os.path.join(utils.fig_path(), utils.get_time_path())
        self.plot_dir += ('_kitti' if self.kitti else '_mine') + f'_{self.endframe}'
        self.keyframes_idx = list(range(0, self.endframe+1, self.bundle_len-1))
        utils.make_dir(self.plot_dir)
        stats = ["tracks_db args:", str(self.tracks_db.args), self.tracks_path]
        with open (os.path.join(self.plot_dir,'stats.txt'), 'w') as f:
            f.writelines('\n'.join(stats))
        

    
    def main(self):
        Pose3_keyframes = gtsam.Values()
        Pose3_keyframes.insert(X(0), Pose3())

        ext_keyframes = []
        for l0_idx in self.keyframes_idx[:-1]:
            endframe = min(l0_idx+self.bundle_len, self.endframe+1)
            if endframe > self.endframe+1:
                break
            lk_to_l0_mine = self.factorize(l0_idx=l0_idx)
            lk_to_l0_kitti = kitti.read_poses_orig([endframe-1])[0]
            l0_to_lk_kitti = kitti.read_poses_world_to_cam([endframe-1])[0]
            Pose3_keyframes.insert(X(endframe-1), lk_to_l0_mine)
            print(f'finished frames [{l0_idx}-{endframe-1}]')
        
        #### plots ####
        # plot 2D camera locations (regular and comparison)
        kitti_dws = kitti.read_dws(idx=self.keyframes_idx) # (3,6)
        my_dws = get_dws_from_gtsam_values(Pose3_keyframes) # (3,6)
        plot_2d_cams(camera_dws=my_dws, plot_dir=self.plot_dir, startframe=0, endframe=self.endframe)
        plot_2d_cams_compare(truth_dws=kitti_dws, my_dws=my_dws, plot_dir=self.plot_dir, endframe=self.endframe)
        # plot trans vecs diff
        kitti_trans_vecs = kitti.read_trans_vectors(idx=self.keyframes_idx)
        my_trans_vecs = get_world_to_cam_trans_vecs_from_values(values=Pose3_keyframes)
        plot_trans_vecs_diff(my_vecs=my_trans_vecs, truth_vecs=kitti_trans_vecs, idx=self.keyframes_idx, plot_dir=self.plot_dir, title="trans")
        # plot 3D Poses
        plot.plot_trajectory(self.endframe, Pose3_keyframes); plot.set_axes_equal(self.endframe)
        plt.savefig(os.path.join(self.plot_dir,f'3d_cams_0_{self.endframe}.png'))


    def factorize(self, l0_idx):
        endframe = l0_idx + self.bundle_len - 1 # 10
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()
        pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 1.0, 1.0, 1.0]))
        # pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001, 0.1, 0.1, 0.1]))
        meas_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

        # add Initial Estimates for Camera
        # initialEstimate.insert(X(l0_idx), Pose3())
        l0_to_li_s = self.tracks_db.ext_l1s[l0_idx:endframe+1]
        li_to_l0_s = [utils.inv_extrinsics(ext_mat) for ext_mat in l0_to_li_s]
        # kitti_li_to_l0_s = kitti.read_poses_orig(list(range(l0_idx, endframe+1)))
        # l0_to_li_s = [np.diag((1, 1, 1, 1))]
        # li_to_l0_s = [np.diag((1, 1, 1, 1))]
        for i in range(l0_idx , endframe+1):
        #     l0_to_li = self.tracks_db.ext_l1s[i] @ l0_to_li_s[-1]
        #     l0_to_li_s.append(l0_to_li)
        #     li_to_l0 = utils.inv_extrinsics(l0_to_li)
        #     li_to_l0_s.append(li_to_l0)
            initialEstimate.insert(X(i), Pose3(li_to_l0_s[i-l0_idx]))

        # add Prior Factor for Pose
        # graph.add( gtsam.PriorFactorPose3(X(l0_idx), Pose3(), pose_noise_model) )
        graph.add( gtsam.PriorFactorPose3(X(l0_idx), Pose3(li_to_l0_s[0]), pose_noise_model) )
        # graph.add( gtsam.NonlinearEqualityPose3(X(l0_idx), Pose3()) )

        
        # TODO should I add prior on some point like in https://github.com/borglab/gtsam/blob/develop/python/gtsam/tests/test_SFMExample.py#L50
        # Add factors for all measurements
        for cam_idx in range(l0_idx, endframe+1):
            idx_tracks = self.tracks_db.get_tracks(cam_id=cam_idx)
            for track in idx_tracks:
                # add factor for measurement
                # [(1026), (30,87), (17,87), 1, array([-22,-3,27])]
                # if we're in endframe, and this track originated in the endframe
                if track.orig_cam_id == endframe:
                    continue
                orig_cam_idx = max(track.orig_cam_id, l0_idx)
                stereo_point = StereoPoint2(track.left_x, track.right_x, track.left_y)
                graph.add( GenericStereoFactor3D(stereo_point, meas_noise_model, X(cam_idx), P(track.id), self.gt_k) )
                # if new point, add initial estimate
                if orig_cam_idx == cam_idx:                    
                    pc_li = track.pc
                    initialEstimate.insert( P(track.id), Point3(pc_li))
                    # pc_in_l0 = li_to_l0_s[cam_idx - l0_idx] @ np.hstack((pc_li,[1]))
                    # initialEstimate.insert( P(track_id), Point3(pc_in_l0[0:3]))

        # optimize
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        values = optimizer.optimize()
        pose_x_endframe = values.atPose3(X(endframe))
        # plot 2D view cameras+points
        plot_2d_cams_points_from_gtsam_values(values=values, plot_dir=self.plot_dir, endframe=endframe, startframe=l0_idx)
        # plot 3D trajectory only cameras
        plot.plot_trajectory(l0_idx, values)
        plot.set_axes_equal(l0_idx)
        plt.savefig(os.path.join(self.plot_dir, f'3d_cams_{l0_idx}_{endframe}'), bbox_inches='tight', pad_inches=0)

        # plot 3D trajectory cameras+points
        plot.plot_trajectory(l0_idx+1, values)
        plot.plot_3d_points(l0_idx+1, values, linespec='r*')
        plot.set_axes_equal(l0_idx+1)
        plt.savefig(os.path.join(self.plot_dir, f'3d_cams_points_{l0_idx}_{endframe}'), bbox_inches='tight', pad_inches=0)
        plt.close('all')
        return pose_x_endframe


if __name__=="__main__":
    tracks_name = '06_08_19_06_mine_global_2760_filtered_2760.pickle'
    # tracks_name = '06_08_18_36_mine_global_50_filtered_50.pickle'
    tracks_path = os.path.join(utils.track_path(), tracks_name)
    ba = FactorGraphSLAM(tracks_path=tracks_path)
    ba.main()
    print('bundle end')