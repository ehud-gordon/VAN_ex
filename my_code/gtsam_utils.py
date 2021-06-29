import pickle

import gtsam
from gtsam import KeyVector
from gtsam.symbol_shorthand import X, P
from gtsam.utils import plot as g_plot
import numpy as np
import matplotlib.pyplot as plt
import utils

import os

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.5g" % x))
#### VALUES ####
def get_dws_from_gtsam_values(values):
    Pose3_values = gtsam.utilities.allPose3s(values)
    cam_to_world_poses = [values.atPose3(k) for k in Pose3_values.keys()]
    cam_to_world_trans = [pose.translation() for pose in cam_to_world_poses]
    cam_to_world_trans = np.array(cam_to_world_trans).T
    return cam_to_world_trans

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

def get_cam_to_world_ext_mats_from_values(values):
    Pose3_values = gtsam.utilities.allPose3s(values)
    cam_to_world_poses = [values.atPose3(k) for k in Pose3_values.keys()]
    cam_to_world_mats = [pose.matrix() for pose in cam_to_world_poses]
    return cam_to_world_mats

def get_Pose3_values_from_cam_to_world_ext_mats(ext_mats, frames_idx):
    assert len(ext_mats) == len(frames_idx)
    Pose3_values = gtsam.Values()
    for mat, frame_idx in zip(ext_mats, frames_idx):
        cur_Pose3 = gtsam.Pose3(mat)
        Pose3_values.insert(X(frame_idx), cur_Pose3)
    return Pose3_values

def get_world_to_cam_rot_trans_from_values(values):
    Pose3_values = gtsam.utilities.allPose3s(values)
    cam_to_world_poses = [values.atPose3(k) for k in Pose3_values.keys()]
    world_to_cam_poses = [pose.inverse() for pose in cam_to_world_poses]
    world_to_cam_rots = np.array([pose.rotation().matrix() for pose in world_to_cam_poses])
    world_to_cam_trans_vecs = [pose.translation() for pose in world_to_cam_poses]
    world_to_cam_trans_vecs = np.array(world_to_cam_trans_vecs).T
    return world_to_cam_rots, world_to_cam_trans_vecs

def get_world_to_cam_ext_from_values(values):
    Pose3_values = gtsam.utilities.allPose3s(values)
    cam_to_world_poses = [values.atPose3(k) for k in Pose3_values.keys()]
    world_to_cam_poses = [pose.inverse() for pose in cam_to_world_poses]
    world_to_cam_exts = [pose.matrix() for pose in world_to_cam_poses]
    return world_to_cam_exts

def r0_to_r1_s_t0_to_t1_s_from_values(values):
    world_to_cam_mats = get_world_to_cam_ext_from_values(values)
    r0_to_r1_s, t0_to_t1_s = utils.r0_to_r1_s_t0_to_t1_s(ext_mats=world_to_cam_mats)
    return r0_to_r1_s, t0_to_t1_s

def serialize_Pose3_marginals(dir_path, values, joint_marginal_cov_mats, relative_cov_mats, frames_idx):
    path = os.path.join(dir_path, f'Pose3_marginals_{frames_idx[-1]}.pkl')
    cam_to_world_mats = get_cam_to_world_ext_mats_from_values(values=values) # ndarray
    assert len(cam_to_world_mats) == len(frames_idx)
    d = dict()
    d['cam_to_world_mats'] = cam_to_world_mats
    d['joint_marginal_cov_mats'] = joint_marginal_cov_mats
    d['relative_cov_mats'] = relative_cov_mats
    d['frames_idx'] = frames_idx
    with open(path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path

def unserialize_Pose3_marginals(pickle_path):
    with open(pickle_path, 'rb') as handle:
        d = pickle.load(handle)
    cam_to_world_mats = d['cam_to_world_mats']
    joint_marginal_cov_mats = d['joint_marginal_cov_mats']
    relative_cov_mats = d['relative_cov_mats']
    frames_idx = d['frames_idx']

    Pose3_values = get_Pose3_values_from_cam_to_world_ext_mats(cam_to_world_mats, frames_idx)
    return Pose3_values, joint_marginal_cov_mats, relative_cov_mats, frames_idx

#### MARGINALS ####
def relative_cov_li_cond_on_l0(marginals, li_idx, l0_idx):
    """ return Sigma li|l0 """
    keys = KeyVector([X(l0_idx), X(li_idx)])
    relative_info_li_cond_on_l0_idx = marginals.jointMarginalInformation(keys).at( X(li_idx), X(li_idx) )
    relative_cov_li_cond_on_l0_idx = np.linalg.inv(relative_info_li_cond_on_l0_idx)
    return relative_cov_li_cond_on_l0_idx

def relative_cov_li_key_cond_on_l0(marginals, li_idx_key, l0_idx):
    """ return Sigma li|l0 """
    keys = KeyVector([X(l0_idx), li_idx_key])
    relative_info_li_cond_on_l0_idx = marginals.jointMarginalInformation(keys).at( li_idx_key, li_idx_key )
    relative_cov_li_cond_on_l0_idx = np.linalg.inv(relative_info_li_cond_on_l0_idx)
    return relative_cov_li_cond_on_l0_idx

def cumsum_mats(mats):
    cumsum_res = [mats[0]]
    for mat in mats[1:]:
        res = cumsum_res[-1] + mat
        cumsum_res.append(res)
    return cumsum_res


#### VISUALIZATION ######
def single_bundle_plots(values, plot_dir, endframe, startframe):
    # plot 2D view cameras+points
    plot_2d_cams_points_from_gtsam_values(values, plot_dir, endframe, startframe)
    
    # plot 3D trajectory only cameras
    gtsam.utils.plot.plot_trajectory(startframe, values)
    gtsam.utils.plot.set_axes_equal(startframe)
    plt.savefig(os.path.join(plot_dir, f'3d_cams_{startframe}_{endframe}'), bbox_inches='tight', pad_inches=0)

    # plot 3D trajectory cameras+points
    gtsam.utils.plot.plot_trajectory(startframe+1, values)
    gtsam.utils.plot.plot_3d_points(startframe+1, values, linespec='r*')
    gtsam.utils.plot.set_axes_equal(startframe+1)
    plt.savefig(os.path.join(plot_dir, f'3d_cams_points_{startframe}_{endframe}'), bbox_inches='tight', pad_inches=0)
    
    plt.close('all')

def plot_2d_cams_points_from_gtsam_values(values, plot_dir, endframe, startframe=0):
    dws = get_dws_from_gtsam_values(values)
    landmarks = get_points_from_gtsam_values(values)
    plt.figure()
    plt.scatter(x=dws[0], y=dws[2], color="red", marker=(5,2), label="camera")
    plt.scatter(x=landmarks[0], y=landmarks[2], color="blue", label="landmark", alpha=0.2)
    plt.xlabel('x'); plt.ylabel('z')
    plt.title(f"2D cameras and landmarks for keyframes [{startframe}-{endframe}]")
    plt.legend()
    path = os.path.join(plot_dir, f'2d_cams_points_{startframe}_{endframe}' + '.png')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)

def my_cond_plot_trajectory(fignum, values, marginals, l0_idx, endframe, plot_dir):
    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')

    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")

    poses = gtsam.utilities.allPose3s(values)
    for key in poses.keys():
        pose = poses.atPose3(key)
        P = relative_cov_li_key_cond_on_l0(marginals, li_idx_key=key, l0_idx=l0_idx)
        if key == X(l0_idx):
            g_plot.plot_pose3_on_axes(axes, pose, axis_length=1)
        else:
            g_plot.plot_pose3_on_axes(axes, pose, P=P, axis_length=1)
        if False: # if we only want to plot the ellipses
            gRp = pose.rotation().matrix()  # rotation from pose to global
            origin = pose.translation()
            pPp = P[3:6, 3:6]
            gPp = gRp @ pPp @ gRp.T
            g_plot.plot_covariance_ellipse_3d(axes, origin, gPp)

    fig.suptitle(f"conditional variance on {l0_idx}, frames [{l0_idx}-{endframe}]")
    # g_plot.set_axes_equal(fignum) # dubious
    path = os.path.join(plot_dir, f'marginal_cov_plot_{l0_idx}_{endframe}')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close('all')