import gtsam
from gtsam.symbol_shorthand import X, P
from gtsam import Pose3, StereoPoint2, GenericStereoFactor3D, Point3
from gtsam.utils import plot
import numpy as np
import matplotlib.pyplot as plt

import os

import tracks
import utils
import kitti
from tracks import MAX_TRCK_PER_IMG
import triang
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.4g" % x))

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

class FactorGraphSLAM:
    def __init__(self, tracks_path):
        self.tracks_path = tracks_path
        self.tracks_db = tracks.read(tracks_path)
        self.endframe = self.tracks_db.endframe # 2760
        self.bundle_len = 10 # number of frames in bundle
        self.k, self.ext_l0, self.ext_r0 = kitti.read_cameras() # k=(3,4) ext_l0/r0 (4,4)
        fx, skew, cx, _, fy, cy = self.k[0:2, 0:3].flatten()
        baseline = self.ext_r0[0, 3]
        self.gt_k = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -baseline)
        self.plot_dir = os.path.join(utils.fig_path(), utils.get_time_path())
        os.makedirs(self.plot_dir)
        stats = [self.tracks_path]
        with open (os.path.join(self.plot_dir,'stats.txt'), 'w') as f:
            f.writelines('\n'.join(stats))
        

    
    def main(self):
        Pose3_keyframes = gtsam.Values()

        ext_keyframes = []
        for l0_idx in range(0,self.endframe, self.bundle_len-1):
            endframe = l0_idx+self.bundle_len
            if endframe > self.endframe:
                break
            lk_to_l0_MY = self.factorize(l0_idx=l0_idx)
            lk_to_l0_pose = kitti.read_poses_orig([endframe-1])[0]
            l0_to_lk_pose = kitti.read_poses_world_to_cam([endframe-1])[0]
            Pose3_keyframes.insert(X(endframe-1), lk_to_l0_MY)
            print(f'finished frames [{l0_idx}-{endframe-1}]')
        plot.plot_trajectory(1, Pose3_keyframes)
        plot.set_axes_equal(1)
        plt.savefig(os.path.join(utils.fig_path(),'ba_3d_poses.png'))
        plt.show()


    def factorize(self, l0_idx):
        endframe = l0_idx + self.bundle_len # 3
        graph = gtsam.NonlinearFactorGraph()
        initialEstimate = gtsam.Values()
        pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001, 0.1, 0.1, 0.1]))
        meas_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

        # add Initial Estimates for Camera
        # initialEstimate.insert(X(l0_idx), Pose3())
        li_to_l0_s = kitti.read_poses_orig(list(range(l0_idx, endframe)))
        #  lim_to_li_s = self.tracks_db.ext_l1s[l0_idx:endframe] # l_{i-1} to l_{i}
        # l0_to_li_s = [np.diag((1, 1, 1, 1))]
        # li_to_l0_s = [np.diag((1, 1, 1, 1))]
        for i in range(l0_idx , endframe):
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
        for cam_idx in range(l0_idx, endframe):
            idx_tracks = self.tracks_db.get_tracks(camera_id=cam_idx)
            for track in idx_tracks:
                # add factor for measurement
                # [(1026), (30,87), (17,87), 1, array([-22,-3,27])]
                track_id = track[0]
                orig_cam_idx, orig_l0_m_idx = max(track_id >> 10, l0_idx), track_id % 1024 # (1, 2)
                if orig_cam_idx >= (endframe-1):
                    continue
                l0_meas, r0_meas = track[1], track[2]
                graph.add( GenericStereoFactor3D(StereoPoint2(l0_meas[0], r0_meas[0], l0_meas[1]),meas_noise_model, X(cam_idx), P(track_id), self.gt_k) )
                # if new point, add initial estimate
                if orig_cam_idx == cam_idx:                    
                    pc_li = track[4]
                    initialEstimate.insert( P(track_id), Point3(pc_li))
                    # pc_in_l0 = li_to_l0_s[cam_idx - l0_idx] @ np.hstack((pc_li,[1]))
                    # initialEstimate.insert( P(track_id), Point3(pc_in_l0[0:3]))

        # optimize
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        values = optimizer.optimize()
        pose_x_endframe = values.atPose3(X(endframe-1))
        
        landmarks = get_points_from_gtsam_values(values)
        landmarks2 = triang.filter_point_cloud(landmarks)
        camera_dws = get_dws_from_gtsam_values(values)
        plt.figure()
        plt.scatter(x=camera_dws[0], y=camera_dws[2], color="red", marker=(5,2), label="camera")
        plt.scatter(x=landmarks2[0], y=landmarks2[2], color="blue", label="landmark", alpha=0.3)
        plt.xlabel('x');plt.ylabel('z')
        plt.title(f"landmarks and cameras for frames [{l0_idx}-{endframe-1}]")
        plt.legend()
        path = os.path.join(self.plot_dir, f'2d_all_{l0_idx}_{endframe-1}' + '.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plot.plot_trajectory(l0_idx, values)
        plot.set_axes_equal(l0_idx)
        plt.savefig(os.path.join(self.plot_dir, f'traj_{l0_idx}_{endframe-1}'), bbox_inches='tight', pad_inches=0)

        plot.plot_trajectory(l0_idx+1, values)
        plot.plot_3d_points(l0_idx+1, values, linespec='r*')
        plot.set_axes_equal(l0_idx+1)
        plt.savefig(os.path.join(self.plot_dir, f'traj_points_{l0_idx}_{endframe-1}'), bbox_inches='tight', pad_inches=0)
        return pose_x_endframe


if __name__=="__main__":
    tracks_path = os.path.join(utils.track_path(), '06_04_23_28_50_pose_global_better_pc.pickle')
    ba = FactorGraphSLAM(tracks_path=tracks_path)
    ba.main()
    print('bundle end')