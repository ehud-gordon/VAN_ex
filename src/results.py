import numpy as np

import os, time
import utils.plot
import utils.geometry
from utils import sys_utils
import kitti


def output_results(out_path, ext_li_to_l0_s_dict, frames_idx,  title, start_time, plot=False, save=True):
    ext_li_to_l0_s_mult = []
    for ext_li_to_l0_s in ext_li_to_l0_s_mult:
        assert len(frames_idx) == len(ext_li_to_l0_s)
    elapsed_time = time.time() - start_time
    # make results dir
    startframe = frames_idx[0]
    endframe = frames_idx[-1]
    res_dir = os.path.join(out_path, 'results')
    sys_utils.make_dir_if_needed(res_dir)
    stats = [f'** {title.upper()} **']
    # kitti
    ext_li_to_l0_s_kitti = kitti.read_poses_cam_to_world(frames_idx)
    rj_to_ri_s_kitti, tj_to_ti_s_kitti = utils.geometry.rot_trans_j_to_i_s(ext_li_to_l0_s_kitti)
    if startframe != 0:
        ext_li_to_l0_s_kitti = utils.geometry.make_relative_to_ci(ext_li_to_l0_s_kitti)
    # compute relative rotation translation diffs
    rot_diffs_rel_mult, trans_diffs_rel_mult = utils.geometry.get_rot_trans_diffs_from_mats(ext_li_to_l0_s_kitti, *ext_li_to_l0_s_mult)
    # normalize by number of frames
    sizes = np.array(frames_idx)[1:] - np.array(frames_idx)[:-1]
    # output relative rots_trans stats and plots
    rot_trans_stats_rel, rots_total_error_rel, trans_total_error_rel = utils.geometry.rot_trans_stats(rot_diffs_relative, trans_diffs_relative, frames_idx, rel_or_abs="relative")
    rot_diffs_relative /= sizes; trans_diffs_relative /= sizes
    utils.plot.plotly_scatter(x=frames_idx[1:], y=rot_diffs_relative, mode="markers+lines", title=f"diff_relative_rot_{title}_{startframe}_{endframe}", yaxis="deg",
                        plot_dir=res_dir, save=save, plot=plot)
    trans_dict = {'tx':trans_diffs_relative[0],'ty':trans_diffs_relative[1], 'tz':trans_diffs_relative[2]}
    utils.plot.plotly_scatters(trans_dict, x=frames_idx[1:], title=f"diff_relative_trans_{title}_{startframe}_{endframe}",
                         plot_dir=res_dir, mode='lines+markers', xaxis="Frames", yaxis="meters", save=save, plot=plot)

    # compute absolute rotation translation diffs
    rot_diffs_abs, trans_diffs_abs = utils.geometry.get_rot_trans_diffs_from_mats(ext_li_to_l0_s[1:], ext_li_to_l0_s_kitti[1:])
    # output absolute rots_trans stats and plots
    rot_trans_stats_abs, rots_total_error_abs, trans_total_error_abs = utils.geometry.rot_trans_stats(rot_diffs_abs, trans_diffs_abs, frames_idx, rel_or_abs="absolute")
    utils.plot.plotly_scatter(x=frames_idx[1:], y=rot_diffs_abs, mode="markers+lines", title=f"diff_absolute_rot_{title}_{startframe}_{endframe}", yaxis="deg",
                        plot_dir=res_dir, save=save, plot=plot)
    trans_dict = {'tx':trans_diffs_abs[0],'ty':trans_diffs_abs[1], 'tz':trans_diffs_abs[2]}
    utils.plot.plotly_scatters(trans_dict, x=frames_idx[1:], title=f"diff_abs_trans_{title}_{startframe}_{endframe}",
                         plot_dir=res_dir, mode='lines+markers', xaxis="Frames", yaxis="meters", save=save, plot=plot)


    my_dws = utils.geometry.get_dws_from_cam_to_world_s(ext_li_to_l0_s)
    kitti_dws = utils.geometry.get_dws_from_cam_to_world_s(ext_li_to_l0_s_kitti)
    comp_title = f'comp_{title}_{startframe}_{endframe}'
    kitti_lsd = [(kitti_dws, 'kitti', 'green')]
    my_lsd = [(my_dws, 'mine', 'red')]
    comp_lsd = [my_lsd[0], kitti_lsd[0]]
    
    utils.plot.plotly_2D_cams(comp_lsd, comp_title, res_dir, frames_idx=frames_idx, save=save, plot=plot)
    utils.plot.plotly_3D_cams(comp_lsd, comp_title, res_dir, frames_idx=frames_idx, save=save, plot=plot)

     # write stats
    stats.append(f'elapsed: {elapsed_time:.0f} sec, avg_per_frame={elapsed_time/endframe:.1f} sec')
    stats += rot_trans_stats_rel + [''] + rot_trans_stats_abs
    stats.append('*************\n')
    print("\n".join(stats))
    if save:
        with open (os.path.join(res_dir, 'stats.txt'), 'a+') as f:
            f.writelines('\n'.join(stats))

    return rots_total_error_abs, trans_total_error_abs

