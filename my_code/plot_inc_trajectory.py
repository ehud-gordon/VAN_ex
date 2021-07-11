import numpy as np
from numpy import pi
import plotly
import gtsam
import matplotlib.pyplot as plt
from gtsam import Pose3
from gtsam.symbol_shorthand import X

from gtsam.utils import plot as g_plot
import os

import my_plot, kitti, utils, pose_graph
import gtsam_utils as g_utils

s3_pkl_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-10-13-49_0_2760/stage3_40.8_2900.9/stage3_ext_lj_to_li_s_cond_covs_2760.pkl'
stage3_dir, _,_ = utils.dir_name_ext(s3_pkl_path)
ext_lj_to_li_s, cov_lj_cond_li_dict, keyframes_idx = g_utils.deserialize_bundle(s3_pkl_path, as_ext=True)
ci_to_c0_s = utils.concat_cj_to_ci_s(ext_lj_to_li_s)
t_dws = utils.get_dws_from_cam_to_world_s(ci_to_c0_s)

Pose3_c_to_w_list = [Pose3(ext) for ext in ci_to_c0_s]
graph = gtsam.NonlinearFactorGraph()
num_frames = len(keyframes_idx) # 277
startframe = keyframes_idx[0]
endframe = keyframes_idx[-1]
cov_cj_cond_ci_s = []
for j in range (1, num_frames):
    cov_cj_cond_ci_s.append(cov_lj_cond_li_dict[j][j-1])
cumsum_cov_cj_cond_ci = utils.cumsum_mats(cov_cj_cond_ci_s)
# Create initial estimate
initialEstimate = gtsam.Values()
for i_kf, ci_to_c0 in zip(keyframes_idx, ci_to_c0_s):
    initialEstimate.insert( X(i_kf), g_utils.to_Pose3(ci_to_c0))

# add prior factor to graph
pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1*pi/180, 1*pi/180, 1*pi/180, 0.3, 0.3, 0.3])) # (1 deg, input must be radian), 0.3 
priorFactor = gtsam.PriorFactorPose3( X(startframe), Pose3() , pose_noise_model)
graph.add(priorFactor)

# add between factors to graph
for j in range(1, len(keyframes_idx)): # [1,2,...,276] represent [10, 20, .. 2760]
    i=j-1
    i_kf, j_kf = keyframes_idx[i], keyframes_idx[j]
    cj_to_c0 = ci_to_c0_s[j]
    ci_to_c0 = ci_to_c0_s[j-1]
    cj_to_ci = utils.B_to_A_mat(ci_to_c0, cj_to_c0)
    cov_cn_cond_ci = cumsum_cov_cj_cond_ci[i]
    noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_cn_cond_ci)
    factor = gtsam.BetweenFactorPose3( X(i_kf), X(j_kf) , g_utils.to_Pose3(cj_to_ci), noise_model)
    graph.add(factor)


def plot_incremental_trajectory(fignum, values, end=0,
                                scale=1, marginals=None,
                                time_interval=0.0):
    """
    Incrementally plot a complete 3D trajectory using poses in `values`.

    Args:
        fignum (int): Integer representing the figure number to use for plotting.
        values (gtsam.Values): Values dict containing the poses.
        start (int): Starting index to start plotting from.
        scale (float): Value to scale the poses by.
        marginals (gtsam.Marginals): Marginalized probability values of the estimation.
            Used to plot uncertainty bounds.
        time_interval (float): Time in seconds to pause between each rendering.
            Used to create animation effect.
    """
    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')
    axes.set_xlabel("X"); axes.set_ylabel("Y"); axes.set_zlabel("Z")

    poses = gtsam.utilities.allPose3s(values)
    keys = gtsam.KeyVector(poses.keys())

    for key in keys[:end]:
        if values.exists(key):
            covariance = marginals.marginalCovariance(key)
            pose_i = values.atPose3(key)
            g_plot.plot_pose3_on_axes(axes, pose_i, P=covariance,
                       axis_length=1)

    # Update the plot space to encompass all plotted points
    axes.autoscale()

    # Set the 3 axes equal
    g_plot.set_axes_equal(fignum)

    # Pause for a fixed amount of seconds
    plt.pause(time_interval)


values, marginals, error_before, error_after = pose_graph.optimize(graph, initialEstimate)
print(f'before_error: {error_before}, error_after: {error_after}')
g_utils.plotly_cond_trajectory(Pose3_c_to_w_list, marginals, cumsum_cov_cj_cond_ci, "pose3", keyframes_idx, title="stage3_pose", plot_dir=stage3_dir,
 save=False, plot=True)
# g_utils.my_cond_plot_trajectory(1, values, marginals,startframe, endframe,stage3_dir)
# for j in range(0,275,30):
#     plot_incremental_trajectory(1, values,end=j, marginals=marginals, time_interval=1)
print('finihsed')

