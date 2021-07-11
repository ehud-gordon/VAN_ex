import numpy as np
import os
import utils, results, my_plot, kitti
from gtsam.symbol_shorthand import X
from collections import defaultdict
import gtsam_utils as g_utils
import pose_graph

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.4g" % x))

def stage4(stg3_pkl_path):
    # init
    stage3_dir, _, _ = utils.dir_name_ext(stg3_pkl_path)
    out_dir = os.path.dirname(stage3_dir)
    stage4_dir = os.path.join(out_dir, 'stage4')
    utils.make_dir_if_needed(stage4_dir)
    # read stage3 pkl
    s3_ext_lj_to_li_s, s3_cov_lj_cond_li_s, keyframes_idx = g_utils.deserialize_bundle(stg3_pkl_path, as_ext=True)
    num_frames = len(keyframes_idx)
    # stage3 extract results
    s3_ext_li_to_l0_s = utils.concat_cj_to_ci_s(s3_ext_lj_to_li_s)
    s3_dws = utils.get_dws_from_cam_to_world_s(s3_ext_li_to_l0_s)
    s3_sd = (s3_dws, 's3', 'red')
    my_plot.plotly_cov_dets(s3_cov_lj_cond_li_s, keyframes_idx, "stage3", stage4_dir, save=True, plot=True)
    # do pose_graph on stage3 restuls
    s3_ext_lj_to_li_dict, s3_cov_lj_cond_li_dict = defaultdict(dict), defaultdict(dict)
    for j in range(1, num_frames):
        s3_ext_lj_to_li_dict[j][j-1] = s3_ext_lj_to_li_s[j]
        s3_cov_lj_cond_li_dict[j][j-1] = s3_cov_lj_cond_li_s[j-1]
    graph, initialEstimate = pose_graph.build_pose_graph(keyframes_idx, s3_ext_li_to_l0_s, s3_ext_lj_to_li_dict, s3_cov_lj_cond_li_dict)
    values, marginals, error_before, error_after = pose_graph.optimize(graph, initialEstimate)
    msg = f'error before pose graph {error_before}\nerror after pose graph {error_after}'; print(msg)
    with open(os.path.join(stage4_dir,'stats_stage4.txt'), 'w') as f: f.write(msg+'\n')
    
    # extract stage4 results
    s4_ext_li_to_l0_s, s4_cov_lj_cond_li_s = pose_graph.extract_ext_cond_from_values_marginals(values, marginals, keyframes_idx)
    s4_pose_dws = utils.get_dws_from_cam_to_world_s(s4_ext_li_to_l0_s)
    # output comparison stage3 and stage4 results, should be the same
    ext_li_to_l0_s_kitti = kitti.read_poses_cam_to_world(keyframes_idx) ;kitti_dws = utils.get_dws_from_cam_to_world_s(ext_li_to_l0_s_kitti)
    kitti_sd = (kitti_dws, 'kitti', 'green') # dws_names_colors
    
    s3_s4_comp_lsd = [s3_sd, (s4_pose_dws,'s4', 'yellow'), kitti_sd]

    my_plot.plotly_2D_cams(s3_s4_comp_lsd, "comp_s3_s4", stage4_dir, frames_idx=keyframes_idx, save=True, plot=True)
    my_plot.plotly_3D_cams(s3_s4_comp_lsd, "comp_s3_s4", stage4_dir, frames_idx=keyframes_idx, save=True, plot=True)
    my_plot.plotly_cov_dets(s4_cov_lj_cond_li_s, keyframes_idx, "stage4", stage4_dir, save=True, plot=True)


if __name__=="__main__":
    stg3_pkl_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-07-11-52_2760/stage3_40.8_29.9/stage3_ext_lj_to_li_s_cond_covs_2760.pkl'
    stage4(stg3_pkl_path)