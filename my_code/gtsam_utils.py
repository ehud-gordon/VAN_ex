import gtsam
from gtsam import KeyVector, Pose3
from gtsam.symbol_shorthand import X
from gtsam.utils import plot as g_plot
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import defaultdict
from gtsam.symbol_shorthand import X
import os, pickle

import utils, my_plot
from utils import und_title

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.5g" % x))

#### GEOMETRY ####
def dws_from_Pose3_c_to_w_s(Pose3_cam_to_world_values):
    """ return dws: (3,n) location of cameras in world coordinates"""
    rot_trans_arr = gtsam.utilities.extractPose3(Pose3_cam_to_world_values)
    dws = rot_trans_arr[:,-3:].T # (3,n)
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

def exts_from_Pose3_values(Pose3_values):
    # Miraculously, this in order
    Pose3_tmp = gtsam.utilities.allPose3s(Pose3_values)
    exts = [Pose3_values.atPose3(k).matrix() for k in Pose3_tmp.keys()]
    return exts

def Pose3_values_from_exts(exts, frames_idx):
    assert len(frames_idx) == len(exts)
    Pose3_values = gtsam.Values()
    for ext, i in zip(exts, frames_idx):
        Pose3_values.insert( X(i), gtsam.Pose3(ext) )
    return Pose3_values

def t2v(pose3):
    rot_mat = pose3.rotation().matrix()    
    trans = pose3.translation()
    return utils.t2v(rot_mat, trans)

def comp_mahal_dist(Pose3_cn_to_ci, cov_cn_cond_ci):
    t2v_n_to_i = t2v(Pose3_cn_to_ci)
    mahal_dist = t2v_n_to_i.T @ cov_cn_cond_ci @ t2v_n_to_i
    return mahal_dist.item()

def rot_trans_norm_from_Pose3(pose):
    rot = pose.rotation().matrix()
    trans = pose.translation()
    return utils.rot_trans_norm(rot, trans)

def get_gt_k(k, ext_l_to_r):
    fx, skew, cx, _, fy, cy = k[0:2, 0:3].flatten()
    baseline = ext_l_to_r[0, 3]
    gt_k = gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -baseline)
    return gt_k

def to_Pose3(cn_to_ci):
    if type(cn_to_ci).__module__ == np.__name__:
        cn_to_ci = Pose3(cn_to_ci)
    return cn_to_ci

def extract_to_c0_from_to_dict(from_to_dict, keyframes_idx, as_ext=True):
    num_frames = len(keyframes_idx)
    cj_to_ci_list = [from_to_dict[0][0]]
    for j in range(1, num_frames):
        cj_to_ci_list.append(from_to_dict[j][j-1])
    
    if type(cj_to_ci_list[0]).__module__ == np.__name__:
        ext_ci_to_c0_s = utils.concat_cj_to_ci_s(cj_to_ci_list)
    else:
        ext_cj_to_ci_s = [pose.matrix() for pose in cj_to_ci_list]
        ext_ci_to_c0_s = utils.concat_cj_to_ci_s(ext_cj_to_ci_s)
    if as_ext:
        return ext_ci_to_c0_s
    else:
        return [Pose3(mat) for mat in ext_ci_to_c0_s]


#### MARGINALS ####
def extract_cov_ln_cond_li_from_marginals(marginals, i_frame, n_frame): # 20, 10
    """ return Sigma n|i """
    keys = KeyVector([X(i_frame), X(n_frame)])
    ln_cond_on_li_idx_info = marginals.jointMarginalInformation(keys).at( X(n_frame), X(n_frame) )
    ln_cond_on_li_idx_cov = np.linalg.inv(ln_cond_on_li_idx_info)
    return ln_cond_on_li_idx_cov

def extract_cov_ln_key_cond_li_from_marginals(marginals, li_idx, ln_idx_key):
    """ return Sigma li|l0 """
    keys = KeyVector( [X(li_idx), ln_idx_key] )
    ln_cond_on_li_info = marginals.jointMarginalInformation(keys).at( ln_idx_key, ln_idx_key )
    ln_cond_on_li_cov = np.linalg.inv(ln_cond_on_li_info)
    return ln_cond_on_li_cov

#### PICKLE ####
def serialize_bundle(dir_path, ext_lj_to_li_s, cov_lj_cond_li_dict, keyframes_idx, title):
    endframe = keyframes_idx[-1]
    pkl_path = os.path.join(dir_path, f'{title}_ext_lj_to_li_s_cond_covs_{endframe}.pkl')
    assert len(keyframes_idx) == len(ext_lj_to_li_s)
    d = dict()
    d['ext_lj_to_li_s'] = ext_lj_to_li_s
    d['cov_lj_cond_li_dict'] = cov_lj_cond_li_dict
    d['keyframes_idx'] = keyframes_idx
    with open(pkl_path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_path

def deserialize_bundle(pickle_path, as_ext=False):
    with open(pickle_path, 'rb') as handle:
        d = pickle.load(handle)
    ext_lj_to_li_s = d['ext_lj_to_li_s']
    cov_lj_cond_li_dict = d['cov_lj_cond_li_dict']
    keyframes_idx = d['keyframes_idx']
    if as_ext:
        return ext_lj_to_li_s, cov_lj_cond_li_dict, keyframes_idx
    Pose3_lj_to_li_list = [Pose3(ext) for ext in ext_lj_to_li_s]
    return Pose3_lj_to_li_list, cov_lj_cond_li_dict, keyframes_idx

def from_to_Pose3_dict(from_to_ext_dict):
    from_to_Pose3_dict = defaultdict(dict)
    for n in from_to_ext_dict:
        for i, ext_ln_to_li in from_to_ext_dict[n].items():
            from_to_Pose3_dict[n][i] = Pose3(ext_ln_to_li)
    return from_to_Pose3_dict

def from_to_ext_dict(from_to_Pose3_dict):
    from_to_ext_dict = defaultdict(dict)
    for n in from_to_Pose3_dict:
        for i, Pose3_ln_to_li in from_to_Pose3_dict[n].items():
            from_to_ext_dict[n][i] = Pose3_ln_to_li.matrix()
    return from_to_ext_dict

def serialize_stage4(dir_path, s4_ext_li_to_l0_s, s4_cov_lj_cond_li_dict, marg_covs, cov_li_cond_l0_s, keyframes_idx, title):
    endframe = keyframes_idx[-1]
    pkl_path = os.path.join(dir_path, f'{title}_{endframe}.pkl')
    d = dict()
    d['s4_ext_li_to_l0_s'] = s4_ext_li_to_l0_s
    d['s4_cov_lj_cond_li_dict'] = s4_cov_lj_cond_li_dict
    d['marg_covs'] = marg_covs
    d['cov_li_cond_l0_s'] = cov_li_cond_l0_s
    d['keyframes_idx'] = keyframes_idx
    with open(pkl_path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_path

def deserialize_stage4(pkl_path):
    with open(pkl_path, 'rb') as handle:
        d = pickle.load(handle)
    s4_ext_li_to_l0_s = d['s4_ext_li_to_l0_s']
    s4_cov_lj_cond_li_dict = d['s4_cov_lj_cond_li_dict']
    marg_covs = d['marg_covs']
    cov_li_cond_l0_s = d['cov_li_cond_l0_s']
    keyframes_idx = d['keyframes_idx']
    return s4_ext_li_to_l0_s, s4_cov_lj_cond_li_dict, marg_covs, cov_li_cond_l0_s, keyframes_idx

def serialize_stage5(dir_path, from_to_Pose3_dict, cov_ln_cond_li_dict, det_ln_cond_li_arr, marg_covs, cov_li_cond_l0_s, keyframes_idx, title):
    endframe = keyframes_idx[-1]
    new_from_to_ext_dict = from_to_ext_dict(from_to_Pose3_dict)
    pkl_path = os.path.join(dir_path, f'{title}_{endframe}.pkl')
    d = dict()
    d['new_from_to_ext_dict'] = new_from_to_ext_dict
    d['cov_ln_cond_li_dict'] = cov_ln_cond_li_dict
    d['det_ln_cond_li_arr'] = det_ln_cond_li_arr
    d['keyframes_idx'] = keyframes_idx
    d['marg_covs'] = marg_covs
    d['cov_li_cond_l0_s'] = cov_li_cond_l0_s
    with open(pkl_path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_path

def deserialize_stage5(pkl_path, as_ext=False, concat=False):
    with open(pkl_path, 'rb') as handle:
        d = pickle.load(handle)
    new_from_to_ext_dict = d['new_from_to_ext_dict']
    cov_ln_cond_li_dict = d['cov_ln_cond_li_dict'] 
    det_ln_cond_li_arr = d['det_ln_cond_li_arr']
    keyframes_idx = d['keyframes_idx']
    marg_covs = d['marg_covs']
    cov_li_cond_l0_s = d['cov_li_cond_l0_s']
    if concat: 
        ext_lj_to_li_s =[new_from_to_ext_dict[0][0]]
        ext_lj_to_li_s += [ new_from_to_ext_dict[j][j-1] for j in range(1,len(keyframes_idx))]
        ext_li_to_l0_s = utils.concat_cj_to_ci_s(ext_lj_to_li_s)
        return ext_li_to_l0_s, cov_ln_cond_li_dict, det_ln_cond_li_arr, marg_covs, cov_li_cond_l0_s, keyframes_idx
    if as_ext:
        return new_from_to_ext_dict, cov_ln_cond_li_dict, det_ln_cond_li_arr, marg_covs, cov_li_cond_l0_s, keyframes_idx
    new_from_to_Pose3_dict = from_to_Pose3_dict(new_from_to_ext_dict)
    return new_from_to_Pose3_dict, cov_ln_cond_li_dict, det_ln_cond_li_arr, marg_covs, cov_li_cond_l0_s, keyframes_idx

#### VISUALIZATION ######
def single_bundle_plots(Pose3_c_to_w_s_points, plot_dir, startframe, endframe, marginals=None):
    # plot 2D view cameras+points
    plot_2d_cams_points_from_gtsam_values(Pose3_c_to_w_s_points, plot_dir, endframe, startframe)
    
    # plot 3D trajectory only cameras
    gtsam.utils.plot.plot_trajectory(startframe, Pose3_c_to_w_s_points)
    gtsam.utils.plot.set_axes_equal(startframe)
    plt.savefig(os.path.join(plot_dir, f'3d_cams_{startframe}_{endframe}'), bbox_inches='tight', pad_inches=0)

    # plot 3D trajectory cameras+points
    gtsam.utils.plot.plot_trajectory(startframe+1, Pose3_c_to_w_s_points)
    gtsam.utils.plot.plot_3d_points(startframe+1, Pose3_c_to_w_s_points, linespec='r*')
    gtsam.utils.plot.set_axes_equal(startframe+1)
    plt.savefig(os.path.join(plot_dir, f'3d_cams_points_{startframe}_{endframe}'), bbox_inches='tight', pad_inches=0)
    if marginals is not None:
        my_cond_plot_trajectory(startframe + 2, Pose3_c_to_w_s_points, marginals, startframe, endframe,
                                        plot_dir)
    plt.close('all')

def plot_2d_cams_points_from_gtsam_values(Pose3_c_to_w_s_points, plot_dir, endframe, startframe=0):
    dws = dws_from_Pose3_c_to_w_s(Pose3_c_to_w_s_points) # (3,n)
    landmarks = points_from_gtsam_values(Pose3_c_to_w_s_points)
    plt.figure()
    plt.scatter(x=dws[0], y=dws[2], color="red", marker=(5,2), label="camera")
    plt.scatter(x=landmarks[0], y=landmarks[2], color="blue", label="landmark", alpha=0.2)
    plt.xlabel('x'); plt.ylabel('z')
    plt.title(f"2D cameras and landmarks for keyframes [{startframe}-{endframe}]")
    plt.legend()
    path = os.path.join(plot_dir, f'2d_cams_points_{startframe}_{endframe}' + '.png')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)

def get_ellipse_trace(ext_cam_to_world, P, name=""):
    """
    :param pose: Pose3 of gtsam, cam to world. 
    :param P: (6,6) covariance matrix of this pose
    This code is copied from gtsam.utils.plot.
    """
    gRp, origin = utils.get_r_t(ext_cam_to_world)
    pPp = P[3:6, 3:6]
    gPp = gRp @ pPp @ gRp.T
    k = 11.82
    U, S, _ = np.linalg.svd(gPp)

    radii = k * np.sqrt(S)
    radii = radii
    rx, ry, rz = radii

    # generate data for "unrotated" ellipsoid
    xc, yc, zc = g_plot.ellipsoid(0, 0, 0, rx, ry, rz, 8)

    # rotate data with orientation matrix U and center c
    data = np.kron(U[:, 0:1], xc) + np.kron(U[:, 1:2], yc) + np.kron(U[:, 2:3], zc)
    n = data.shape[1]
    x = data[0:n, :] + origin[0]
    y = data[n:2*n, :] + origin[1]
    z = data[2*n:, :] + origin[2]

    ellipse_trace = go.Surface(x=x, y=z, z=y, opacity=0.5, showscale=False, showlegend=(name != ""), name=name)
    return ellipse_trace

def plotly_cond_trajectory2(ext_ci_to_c0_s, cumsum_cov_cj_cond_ci, name, frames_idx, title="", plot_dir="", save=True, plot=False):
    dws= utils.get_dws_from_cam_to_world_s(ext_ci_to_c0_s)
    num_frames = dws.shape[1] # 11
    startframe = frames_idx[0]
    frames_idx = np.array(frames_idx) if frames_idx is not None else np.arange(0,num_frames)  # [0-10]

    fig = go.Figure()
    # create fixed scatter
    trace = go.Scatter3d(x=dws[0], y=dws[2], z=dws[1], name=name, mode='markers+lines', line=dict(color='green'), marker=dict(size=3.5, color='green', opacity=0.5), hovertemplate="(%{x:.1f}, %{z:.1f}, %{y:.1f}) f=%{text}", text=frames_idx)
    # trace = go.Scatter3d(x=dws[0], y=dws[1], z=dws[2], name=name, mode='markers+lines', line=dict(color='green'), marker=dict(size=3.5, color='green', opacity=0.5), hovertemplate="(%{x:.1f}, %{y:.1f}, %{z:.1f}) f=%{text}", text=frames_idx)
    
    fig.add_trace(trace)
    camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.2, y=-2, z=0.5)); fig.update_layout(scene_camera=camera)
    
    # add elipsoids
    for j in range(1, len(ext_ci_to_c0_s), 2):
        ext_lj_to_l0 = ext_ci_to_c0_s[j]
        P= cumsum_cov_cj_cond_ci[j-1]

        ellipse_trace = get_ellipse_trace(ext_lj_to_l0, P)
        fig.add_trace(ellipse_trace)
        
    
    # fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), width=850, height=850)
    fig.update_layout(showlegend=True)
    title1 = f"left camera location {title}"
    fig.update_layout(title_text=title1,  title_x=0.5, font=dict(size=14))
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Z', zaxis_title='Y', aspectmode='data')); fig.update_scenes(zaxis_autorange="reversed")
    my_plot.plotly_save_fig(fig, f'cond_traj_plotly_data_{title}', plot_dir, save, plot)
    fig.update_layout(scene=dict(aspectmode='cube'))
    my_plot.plotly_save_fig(fig, f'cond_traj_plotly_cube_{title}', plot_dir, save, plot)

def plotly_cond_trajectory(Pose3_c_to_w_list, marginals, cumsum_cov_cj_cond_ci, name, frames_idx, title="", plot_dir="", save=True, plot=False):
    ext_c_to_w_s = [pose.matrix() for pose in Pose3_c_to_w_list]
    dws= utils.get_dws_from_cam_to_world_s(ext_c_to_w_s)
    num_frames = dws.shape[1] # 11
    startframe = frames_idx[0]
    frames_idx = np.array(frames_idx) if frames_idx is not None else np.arange(0,num_frames)  # [0-10]

    fig = go.Figure()
    # create fixed scatter
    trace = go.Scatter3d(x=dws[0], y=dws[2], z=dws[1], name=name, mode='markers+lines', line=dict(color='green'), marker=dict(size=3.5, color='green', opacity=0.5), hovertemplate="(%{x:.1f}, %{z:.1f}, %{y:.1f}) f=%{text}", text=frames_idx)
    # trace = go.Scatter3d(x=dws[0], y=dws[1], z=dws[2], name=name, mode='markers+lines', line=dict(color='green'), marker=dict(size=3.5, color='green', opacity=0.5), hovertemplate="(%{x:.1f}, %{y:.1f}, %{z:.1f}) f=%{text}", text=frames_idx)
    
    fig.add_trace(trace)
    camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.2, y=-2, z=0.5)); fig.update_layout(scene_camera=camera)
    
    # add elipsoids
    print(len(Pose3_c_to_w_list))
    for j in range(1, len(Pose3_c_to_w_list), 30):
        j_kf = frames_idx[j]
        pose = Pose3_c_to_w_list[j]
        P = marginals.marginalCovariance( X(j_kf) )
        P = extract_cov_ln_cond_li_from_marginals(marginals, 0, j_kf)
        # P= cumsum_cov_cj_cond_ci[j-1]

        ellipse_trace = get_ellipse_trace(pose, P)
        fig.add_trace(ellipse_trace)
        
    
    # fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), width=850, height=850)
    fig.update_layout(showlegend=True)
    title1 = f"left camera location {title}"
    fig.update_layout(title_text=title1,  title_x=0.5, font=dict(size=14))
    # fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube'))
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Z', zaxis_title='Y', aspectmode='cube')); fig.update_scenes(zaxis_autorange="reversed")
    my_plot.plotly_save_fig(fig, f'cond_traj_plotly_{title}', plot_dir, save, plot)

def plt_plot_cov_trajectory(fignum, Pose3_c_to_w_list, cov_list, frames_idx, title="", plot_dir="", save=True, plot=False):
    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')
    startframe = frames_idx[0]; endframe= frames_idx[-1]
    num_frames = len(frames_idx)
    axes.set_xlabel("X"); axes.set_ylabel("Y");axes.set_zlabel("Z")
    for i in range(0, num_frames-1):
        # if i %20 != 0: continue
        pose = Pose3_c_to_w_list[i]
        P = cov_list[i]
        if i == 0:
            g_plot.plot_pose3_on_axes(axes, pose, axis_length=1)
        else:
            g_plot.plot_pose3_on_axes(axes, pose, P=P, axis_length=1)

    fig.suptitle(f"{title} ellipses of covariance conditional on {startframe}, between [{startframe}-{endframe}]")
    g_plot.set_axes_equal(fignum) # dubious

    if save:
        path = os.path.join(plot_dir, f"elips_cov{und_title(title)}{endframe}.png")
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()

    # plt.close('all')

def my_cond_plot_trajectory(fignum, Pose3_c_to_w_s, marginals, startframe, endframe, plot_dir):
    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')

    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")

    poses = gtsam.utilities.allPose3s(Pose3_c_to_w_s)
    for j, key in enumerate(poses.keys()):
        if j %20 != 0:
            continue
        pose = poses.atPose3(key)

        P = extract_cov_ln_key_cond_li_from_marginals(marginals, ln_idx_key=key, li_idx=startframe)
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

    fig.suptitle(f"ellipses of covariance conditional on {startframe}, bundle between [{startframe}-{endframe}]")
    # g_plot.set_axes_equal(fignum) # dubious
    path = os.path.join(plot_dir, f'marginal_cov_plot_{startframe}_{endframe}')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()    
    # plt.close('all')

if __name__=="__main__":
    pass