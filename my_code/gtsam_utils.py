import gtsam
from gtsam import KeyVector, Pose3
from gtsam.symbol_shorthand import X
from gtsam.utils import plot as g_plot
import numpy as np
import matplotlib.pyplot as plt

import os, pickle

import utils

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.5g" % x))

#### GEOMETRY ####
def dws_from_gtsam_values(Pose3_cam_to_world_values):
    """ return dws: (3,n) location of cameras in world coordinates"""
    rot_trans_arr = gtsam.utilities.extractPose3(Pose3_cam_to_world_values)
    dws = rot_trans_arr[:,-3:].T
    return dws

def points_from_gtsam_values(values):
    points = []
    for k in values.keys():
        try:
            p =values.atPoint3(k)
            points.append(p)
        except RuntimeError:
            continue
    points = np.array(points).T
    return points

def ext_ci_to_c0_s_from_values(Pose3_ci_to_c0_values):
    # Miraculously, this in order
    Pose3_ci_to_c0_s = gtsam.utilities.allPose3s(Pose3_ci_to_c0_values)
    ext_ci_to_c0_s = [Pose3_ci_to_c0_values.atPose3(k).matrix() for k in Pose3_ci_to_c0_s.keys()]
    return ext_ci_to_c0_s

def Pose3_ci_to_c0_s_from_ext_ci_to_c0_s(ext_ci_to_c0_s, frames_idx):
    assert len(ext_ci_to_c0_s) == len(frames_idx)
    Pose3_ci_to_c0_s = gtsam.Values()
    for ext_ci_to_c0, frame_idx in zip(ext_ci_to_c0_s, frames_idx):
        Pose3_ci_to_c0_s.insert( X(frame_idx), gtsam.Pose3(ext_ci_to_c0) )
    return Pose3_ci_to_c0_s

def ext_c0_to_ci_s_from_values(values):
    Pose3_values = gtsam.utilities.allPose3s(values)
    Pose3_ci_to_c0_s = [values.atPose3(k) for k in Pose3_values.keys()]
    ext_c0_to_ci_s = [pose.inverse().matrix() for pose in Pose3_ci_to_c0_s]
    return ext_c0_to_ci_s

def t2v(pose3):
    rot_mat = pose3.rotation().matrix()    
    trans = pose3.translation()
    return utils.t2v(rot_mat, trans)

def Pose3_cn_to_ci(Pose3_cn_to_c0, Pose3_ci_to_c0):
    ext_ci_to_c0 = Pose3_ci_to_c0.matrix()
    ext_cn_to_c0 = Pose3_cn_to_c0.matrix()
    ext_cn_to_ci = utils.B_to_A_mat(ext_ci_to_c0, ext_cn_to_c0)
    Pose3_cn_to_ci = Pose3(ext_cn_to_ci)
    return Pose3_cn_to_ci

def rot_trans_norm_from_Pose3(pose):
    rot = pose.rotation().matrix()
    trans = pose.translation()
    return utils.rot_trans_norm(rot, trans)

def get_gt_k(k, ext_l_to_r):
    fx, skew, cx, _, fy, cy = k[0:2, 0:3].flatten()
    baseline = ext_l_to_r[0, 3]
    gt_k = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -baseline)
    return gt_k

#### MARGINALS ####
def cov_ln_cond_on_li(marginals, n_frame, i_frame): # 20, 10
    """ return Sigma n|i """
    keys = KeyVector([X(i_frame), X(n_frame)])
    ln_cond_on_li_idx_info = marginals.jointMarginalInformation(keys).at( X(n_frame), X(i_frame) )
    ln_cond_on_li_idx_cov = np.linalg.inv(ln_cond_on_li_idx_info)
    return ln_cond_on_li_idx_cov

def cov_ln_key_cond_on_li(marginals, ln_idx_key, li_idx):
    """ return Sigma li|l0 """
    keys = KeyVector( [X(li_idx), ln_idx_key] )
    ln_cond_on_li_info = marginals.jointMarginalInformation(keys).at( ln_idx_key, ln_idx_key )
    ln_cond_on_li_cov = np.linalg.inv(ln_cond_on_li_info)
    return ln_cond_on_li_cov

def cov_lj_cond_on_li_s(marginals, frames_idx):
    cov_lj_cond_on_li_s = []
    for j_kf, i_kf in zip (frames_idx[1:], frames_idx[:-1]):
        cov_lj_cond_on_li = cov_ln_cond_on_li(marginals, j_kf, i_kf)
        cov_lj_cond_on_li_s.append(cov_lj_cond_on_li)
    cov_li_cond_l0_cumsum = cumsum_mats(cov_lj_cond_on_li_s)
    cov_li_cond_l0_cumsum.insert(0, np.zeros((6,6))) # (277,) a[i]= Sigma_i|0
    
    return cov_lj_cond_on_li_s, cov_li_cond_l0_cumsum

def cumsum_mats(mats):
    cumsum_res = [mats[0]]
    for mat in mats[1:]:
        res = cumsum_res[-1] + mat
        cumsum_res.append(res)
    return cumsum_res

#### PICKLE ####
def serialize_bundle(dir_path, Pose3_li_to_l0_keyframes, cov_lj_cond_li_keyframes, keyframes_idx, title):
    path = os.path.join(dir_path, f'{title}_Pose3_marginals_{keyframes_idx[-1]}.pkl')
    ext_li_to_l0_s = ext_ci_to_c0_s_from_values(Pose3_li_to_l0_keyframes)
    assert len(ext_li_to_l0_s) == len(keyframes_idx)
    d = dict()
    d['ext_li_to_l0_s'] = ext_li_to_l0_s
    d['cov_lj_cond_li_keyframes'] = cov_lj_cond_li_keyframes
    d['keyframes_idx'] = keyframes_idx
    with open(path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path

def unserialize_bundle(pickle_path):
    with open(pickle_path, 'rb') as handle:
        d = pickle.load(handle)
    ext_li_to_l0_s = d['ext_li_to_l0_s']
    cov_lj_cond_li_keyframes = d['cov_lj_cond_li_keyframes']
    keyframes_idx = d['keyframes_idx']
    Pose3_li_to_l0_s = Pose3_ci_to_c0_s_from_ext_ci_to_c0_s(ext_li_to_l0_s, keyframes_idx)
    return Pose3_li_to_l0_s, cov_lj_cond_li_keyframes, keyframes_idx

#### VISUALIZATION ######
def single_bundle_plots(values, plot_dir, startframe, endframe):
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
    dws = dws_from_gtsam_values(values)
    landmarks = points_from_gtsam_values(values)
    plt.figure()
    plt.scatter(x=dws[0], y=dws[2], color="red", marker=(5,2), label="camera")
    plt.scatter(x=landmarks[0], y=landmarks[2], color="blue", label="landmark", alpha=0.2)
    plt.xlabel('x'); plt.ylabel('z')
    plt.title(f"2D cameras and landmarks for keyframes [{startframe}-{endframe}]")
    plt.legend()
    path = os.path.join(plot_dir, f'2d_cams_points_{startframe}_{endframe}' + '.png')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)

def my_cond_plot_trajectory(fignum, values, marginals, startframe, endframe, plot_dir):
    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')

    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")

    poses = gtsam.utilities.allPose3s(values)
    for key in poses.keys():
        pose = poses.atPose3(key)
        P = cov_ln_key_cond_on_li(marginals, ln_idx_key=key, li_idx=startframe)
        if key == X(startframe):
            g_plot.plot_pose3_on_axes(axes, pose, axis_length=1)
        else:
            g_plot.plot_pose3_on_axes(axes, pose, P=P, axis_length=1)
        if False: # if we only want to plot the ellipses
            gRp = pose.rotation().matrix()  # rotation from pose to global
            origin = pose.translation()
            pPp = P[3:6, 3:6]
            gPp = gRp @ pPp @ gRp.T
            g_plot.plot_covariance_ellipse_3d(axes, origin, gPp)

    fig.suptitle(f"ellipses of covariance conditional on {startframe}, frames [{startframe}-{endframe}]")
    # g_plot.set_axes_equal(fignum) # dubious
    path = os.path.join(plot_dir, f'marginal_cov_plot_{startframe}_{endframe}')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close('all')