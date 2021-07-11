import numpy as np
from numpy import pi
from scipy.sparse.csr import csr_matrix
from scipy.sparse.csgraph import dijkstra
import gtsam
from gtsam import Pose3
from gtsam.symbol_shorthand import X
import gtsam_utils as g_utils

import os
from collections import defaultdict

import utils, results, my_plot, kitti, features, pnp, stam2, pose_graph
import shortest_path

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.4g" % x))
    

# s3
stg3_pkl_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-10-13-49_0_2760/stage3_40.8_29.9/stage3_ext_lj_to_li_s_cond_covs_2760.pkl'    # init
stage3_dir, _, _ = utils.dir_name_ext(stg3_pkl_path)
out_dir = os.path.dirname(stage3_dir)
# read stage3 pkl
s3_ext_lj_to_li_s, cov_ln_cond_li_dict, keyframes_idx = g_utils.deserialize_bundle(stg3_pkl_path, as_ext=True)
num_frames = len(keyframes_idx)
# kitti
kitti_ext_li_to_l0_s = kitti.read_poses_cam_to_world(keyframes_idx)
kitti_dws = utils.get_dws_from_cam_to_world_s(kitti_ext_li_to_l0_s)
kitti_sd = (kitti_dws, 'kitti', 'green')
# create s3 checkpoint
s3_ext_li_to_l0_s = utils.concat_cj_to_ci_s(s3_ext_lj_to_li_s)
s3_dws = utils.get_dws_from_cam_to_world_s(s3_ext_li_to_l0_s); s3_sd = (s3_dws, 's3', 'red')
# stage3 to lc format
from_to_Pose3_dict = defaultdict(dict)
from_to_Pose3_dict[0][0] = Pose3(s3_ext_lj_to_li_s[0])
det_ln_cond_li_arr = np.zeros( (num_frames, num_frames) )
for j in range(1, num_frames): #[1,..,276]
    i=j-1 # [0,..,275]
    j_kf = keyframes_idx[j] 
    from_to_Pose3_dict[j][i] = Pose3(s3_ext_lj_to_li_s[j])
    det_ln_cond_li_arr[j,i] = np.linalg.det( cov_ln_cond_li_dict[j][i] )
det_ln_cond_li_arr = csr_matrix(det_ln_cond_li_arr)

# ADD LOOP_CLOSURE
Pose3_l1330_to_l570, cov_l1330_cond_l570, cov_l570_cond_l1330, det_ln_cond_li = stam2.add_edges(570, 1330)
from_to_Pose3_dict[133][57] = Pose3_l1330_to_l570
cov_ln_cond_li_dict[133][57] = cov_l1330_cond_l570
cov_ln_cond_li_dict[57][133] = cov_l570_cond_l1330
det_ln_cond_li_arr[133,57] = det_ln_cond_li


######## BUILD POSE GRAPH WITH DIJKSTRA!!
det_dists_ln_cond_li, pred_to_n = dijkstra(det_ln_cond_li_arr, directed=False, return_predecessors=True)

# BUILD DWS WITH DIJKSTRA
dijk_ext_li_to_l0_s = [np.diag([1,1,1,1])]
dijk_cov_li_on_l0_s = [np.zeros((6,6))]
dijk_det_li_on_l0_s = [0]
num_steps_to_zero = [0]
for n in range(1, num_frames): # [1,...,276]
    # find ln_to_l0_s
    Pose3_ln_to_l0, cov_ln_cond_l0, simp_path = shortest_path.Pose3_and_cov_ln_to_li_from_pred(0, n, from_to_Pose3_dict, cov_ln_cond_li_dict, pred_to_n[n])
    dijk_ext_li_to_l0_s.append(Pose3_ln_to_l0.matrix())
    dijk_cov_li_on_l0_s.append(cov_ln_cond_l0)
    dijk_det_li_on_l0_s.append(np.linalg.det(cov_ln_cond_l0))
    num_steps_to_zero.append(len(simp_path)-1 )

dijk_dws = utils.get_dws_from_cam_to_world_s(dijk_ext_li_to_l0_s)
dijk_sd = (dijk_dws, 'dijk_dws', 'orange')
# my_plot.plotly_scatter(x=keyframes_idx, y=dijk_det_li_on_l0_s, yaxis='det of cov', title="dijk_li_on_l0_s", save=True, plot=True)
# my_plot.plotly_scatter(x=keyframes_idx, y=num_steps_to_zero, yaxis='num of steps to zero', title="path_to_0_len", save=True, plot=True)
# my_plot.plotly_3D_cams([s3_sd, dijk_sd, kitti_sd], title="with_dijk", plot_dir=out_dir, frames_idx=keyframes_idx, save=True, plot=True)


startframe = keyframes_idx[0]
graph = gtsam.NonlinearFactorGraph()

# Create initial estimate - ci_to_c0
initialEstimate = gtsam.Values()
for i, dijk_li_to_l0 in enumerate (dijk_ext_li_to_l0_s): # [0,..,276]
    i_kf = keyframes_idx[i]
    initialEstimate.insert( X(i_kf), g_utils.to_Pose3(dijk_li_to_l0) )

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
    cov_cj_cond_c0 = dijk_cov_li_on_l0_s[j]
    noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_cj_cond_c0)
    factor = gtsam.BetweenFactorPose3( X(i_kf), X(j_kf) , g_utils.to_Pose3(cj_to_ci), noise_model)
    graph.add(factor)

noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_l1330_cond_l570)
factor = gtsam.BetweenFactorPose3( X(570), X(1330) , g_utils.to_Pose3(Pose3_l1330_to_l570), noise_model)
graph.add(factor)

values, marginals, error_before, error_after = pose_graph.optimize(graph, initialEstimate)
print(f'dijk init estimate pose_graph error_before: {error_before}, error_after: {error_after}')
after_pose_ci_to_c0_s = pose_graph.extract_ext_ci_to_c0_s_from_values(values, keyframes_idx)
after_pose_dws = utils.get_dws_from_cam_to_world_s(after_pose_ci_to_c0_s)
after_pose_sd = (after_pose_dws, 'after_pose', 'pink')


# output results
my_plot.plotly_3D_cams([s3_sd, dijk_sd ,after_pose_sd, kitti_sd], title="dijk+after_pose", plot_dir=out_dir, frames_idx=keyframes_idx, save=True, plot=True)
# results.output_results(out_dir, s3_init_ci_to_c0_s, frames_idx=keyframes_idx, title="stage5_lc_1_s3_init", start_time=0, plot=True, save=True)
