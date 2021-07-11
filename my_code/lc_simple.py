import os

import numpy as np
import os
from numpy import pi
import utils, results, my_plot, kitti, features, pnp, stam2
import gtsam
from gtsam import Pose3
from gtsam.symbol_shorthand import X
from collections import defaultdict
import gtsam_utils as g_utils
import pose_graph

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.4g" % x))

# s3
stg3_pkl_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-10-13-49_0_2760/stage3_40.8_29.9/stage3_ext_lj_to_li_s_cond_covs_2760.pkl'    # init
stage3_dir, _, _ = utils.dir_name_ext(stg3_pkl_path)
out_dir = os.path.dirname(stage3_dir)
# read stage3 pkl
s3_ext_lj_to_li_s, s3_cov_lj_cond_li_s, keyframes_idx = g_utils.deserialize_bundle(stg3_pkl_path)
num_frames = len(keyframes_idx)
# stage3 extract results
s3_ext_li_to_l0_s = utils.concat_cj_to_ci_s(s3_ext_lj_to_li_s)
s3_ext_l570_to_l0 = s3_ext_li_to_l0_s[57]

s3_dws = utils.get_dws_from_cam_to_world_s(s3_ext_li_to_l0_s)
s3_sd = (s3_dws, 's3', 'red')
# kitti
kitti_ext_li_to_l0_s = kitti.read_poses_cam_to_world(keyframes_idx)
kitti_dws = utils.get_dws_from_cam_to_world_s(kitti_ext_li_to_l0_s)
kitti_sd = (kitti_dws, 'kitti', 'green')

my_plot.plotly_3D_cams([s3_sd, kitti_sd], title="tmp", frames_idx=keyframes_idx, save=False, plot=True)
exit()
s3_cov_lj_cond_l0_s = utils.cumsum_mats(s3_cov_lj_cond_li_s)
# s3_ext_lj_to_li_dict, s3_cov_lj_cond_li_dict = defaultdict(dict), defaultdict(dict)
# for j in range(1, num_frames):
#     s3_ext_lj_to_li_dict[j][j-1] = s3_ext_lj_to_li_s[j]
#     s3_cov_lj_cond_li_dict[j][j-1] = s3_cov_lj_cond_li_s[j-1]

######## BUILD POSE GRAPH
graph = gtsam.NonlinearFactorGraph()
startframe = keyframes_idx[0]

# Create initial estimate
initialEstimate = gtsam.Values()
for i_kf, ci_to_c0 in zip(keyframes_idx, s3_ext_li_to_l0_s):
    initialEstimate.insert( X(i_kf), g_utils.to_Pose3(ci_to_c0))

# add prior factor to graph
pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1*pi/180, 1*pi/180, 1*pi/180, 0.3, 0.3, 0.3])) # (1 deg, input must be radian), 0.3 
priorFactor = gtsam.PriorFactorPose3( X(startframe), Pose3() , pose_noise_model)
graph.add(priorFactor)

# add between factors to graph
for j in range(1, len(keyframes_idx)): # [1,2,...,276] represent [10, 20, .. 2760]
    i=j-1
    i_kf, j_kf = keyframes_idx[i], keyframes_idx[j]
    ci_to_c0 = s3_ext_li_to_l0_s[i]
    cj_to_c0 = s3_ext_li_to_l0_s[j]
    cj_to_ci = utils.B_to_A_mat(ci_to_c0, cj_to_c0)
    cov_cj_cond_c0 = s3_cov_lj_cond_l0_s[i]
    noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_cj_cond_c0)
    factor = gtsam.BetweenFactorPose3( X(i_kf), X(j_kf) , g_utils.to_Pose3(cj_to_ci), noise_model)
    graph.add(factor)

Pose3_l1330_to_l570, cov_l1330_cond_l570 = stam2.add_edges(570, 1330)

noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_l1330_cond_l570)
factor = gtsam.BetweenFactorPose3( X(570), X(1330) , g_utils.to_Pose3(Pose3_l1330_to_l570), noise_model)
graph.add(factor)

values, marginals, errors_before, error_after = pose_graph.optimize(graph, initialEstimate)

after_pose_ci_to_c0_s = pose_graph.extract_ext_ci_to_c0_s_from_values(values, keyframes_idx)
after_pose_dws = utils.get_dws_from_cam_to_world_s(after_pose_ci_to_c0_s)
after_pose_sd = (after_pose_dws, 'after_pose', 'orange')


# output results
results.output_results(out_dir, after_pose_ci_to_c0_s,frames_idx=keyframes_idx, title="stage5_lc_1", start_time=0, plot=True, save=True)
# my_plot.plotly_3D_cams([s3_sd, after_pose_sd, kitti_sd], title="after_pose", plot_dir=out_dir, frames_idx=keyframes_idx, save=True, plot=True)