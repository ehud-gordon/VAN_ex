import numpy as np
import matplotlib.pyplot as plt

import os, time

import utils, kitti, my_plot


def output_results(out_path, ext_l0_to_l1_s, frames_idx,  title, start_time,relative, plot=False, save=True):
    elapsed_time = time.time() - start_time
    # make results dir 
    endframe = frames_idx[-1]
    res_dir = os.path.join(out_path,'results')
    res_dir = utils.get_avail_path(res_dir) # TODO remove
    utils.make_dir_if_needed(res_dir)
    stats = [f'** {title.upper()} **']
    # compute relative rotation translation diffs
    ext_l0_to_l1_s_kitti = kitti.read_poses_world_to_cam(idx=frames_idx)
    if relative:
        r0_to_r1_s_mine, t0_to_t1_s_mine = utils.r0_to_r1_s_t0_to_t1_s_from_relative(ext_l0_to_l1_s[1:])
    else:
        r0_to_r1_s_mine, t0_to_t1_s_mine = utils.r0_to_r1_s_t0_to_t1_s(ext_l0_to_l1_s)
    r0_to_r1_s_kitti, t0_to_t1_s_kitti = utils.r0_to_r1_s_t0_to_t1_s(ext_l0_to_l1_s_kitti)
    rot_diffs_relative, trans_diffs_relative  = utils.get_rot_trans_diffs(r0_to_r1_s_mine, r0_to_r1_s_kitti, t0_to_t1_s_mine, t0_to_t1_s_kitti )
    rot_trans_stats, rots_total_error, trans_total_error = utils.rot_trans_stats(rot_diffs_relative, trans_diffs_relative, endframe)

     # plot relative rotation translation diffs
    my_plot.plt_diff_rot_matrices(rot_diffs_relative, res_dir, title, frames_idx[1:], plot=plot, save=save)
    my_plot.plt_diff_trans_vecs(trans_diffs_relative, res_dir, title, frames_idx[1:], plot=plot, save=save)


    # plot camera locations
    if relative:
        global_ext_mats = utils.relative_to_global(ext_l0_to_l1_s)
        my_dws = utils.get_dws_from_extrinsics(global_ext_mats)
    else:
        my_dws = utils.get_dws_from_extrinsics(ext_l0_to_l1_s)
    kitti_dws = kitti.read_dws(idx=frames_idx) # (3,6)
    my_plot.plt_2d_cams_compare(my_dws, kitti_dws, res_dir, title, endframe=endframe, plot=plot, save=save)
    my_plot.plt_3d_cams_compare(my_dws, kitti_dws, res_dir, title, endframe=endframe, plot=plot, save=save)
    my_plot.plotly_2d_cams_compare(my_dws, kitti_dws, res_dir, title, frames_idx=frames_idx, save=save)
    my_plot.plotly_3d_cams_compare(my_dws, kitti_dws, res_dir, title, frames_idx=frames_idx, save=save)

     # write stats
    stats.append(f'elapsed: {elapsed_time:.0f} sec, avg_per_frame={elapsed_time/endframe:.1f} sec')
    stats += rot_trans_stats
    stats.append('*************\n')
    print("\n".join(stats))
    with open (os.path.join(res_dir, 'stats.txt'), 'a+') as f:
        f.writelines('\n'.join(stats))


    return rots_total_error, trans_total_error

