import numpy as np
import matplotlib.pyplot as plt

import os, time

import utils, kitti, my_plot
np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.5g" % x))


def output_results(out_path, ext_li_to_l0_s, frames_idx,  title, start_time, relative, plot=False, save=True):
    elapsed_time = time.time() - start_time
    # make results dir 
    startframe = frames_idx[0]
    endframe = frames_idx[-1]
    res_dir = os.path.join(out_path, 'results')
    utils.make_dir_if_needed(res_dir)
    stats = [f'** {title.upper()} **']
    # compute relative rotation translation diffs
    ext_li_to_l0_s_kitti = kitti.read_poses_cam_to_world(idx=frames_idx)
    if relative:
        pass  #TODO IMPLEMENT
    else:
        rj_to_ri_s_mine, tj_to_ti_s_mine = utils.rot_trans_j_to_i_s(ext_li_to_l0_s)
    rj_to_ri_s_kitti, tj_to_ti_s_kitti = utils.rot_trans_j_to_i_s(ext_li_to_l0_s_kitti)
    rot_diffs_relative, trans_diffs_relative = utils.get_rot_trans_diffs( rj_to_ri_s_mine, rj_to_ri_s_kitti, tj_to_ti_s_mine, tj_to_ti_s_kitti )
    rot_trans_stats, rots_total_error, trans_total_error = utils.rot_trans_stats(rot_diffs_relative, trans_diffs_relative, endframe)

     # plot relative rotation translation diffs
    my_plot.plt_diff_rot_matrices(rot_diffs_relative, res_dir, title, frames_idx[1:], plot=plot, save=save)
    my_plot.plt_diff_trans_vecs(trans_diffs_relative, res_dir, title, frames_idx[1:], plot=plot, save=save)


    # plot camera locations
    if relative:
        pass # TODO IMPLEMENT
    else:
        my_dws = utils.get_dws_from_cam_to_world_s(ext_li_to_l0_s)
    my_title = f'mine_{title}_{endframe}'
    kitti_title = f'kitti_{endframe}'
    comp_title = f'comp_{title}_{endframe}'
    kitti_dws = utils.get_dws_from_cam_to_world_s(ext_li_to_l0_s_kitti)
    kitti_lsd = [(kitti_dws, 'kitti', 'green')]
    my_lsd = [(my_dws, 'mine', 'red')]
    comp_lsd = [my_lsd[0], kitti_lsd[0]]
    
    # my_plot.plt_2D_cams(my_lsd, my_title, res_dir, plot=plot, save=save)
    my_plot.plt_3D_cams(comp_lsd, comp_title, res_dir, plot=plot, save=save)
    
    # my_plot.plotly_2D_cams(kitti_lsd, kitti_title, res_dir, frames_idx=frames_idx)
    # my_plot.plotly_2D_cams(my_lsd, my_title, res_dir, frames_idx=frames_idx)
    my_plot.plotly_2D_cams(comp_lsd, comp_title, res_dir, frames_idx=frames_idx, plot=plot, save=save)
    
    # my_plot.plotly_3D_cams(kitti_lsd, kitti_title, res_dir, frames_idx=frames_idx, plot=plot, save=save)
    # my_plot.plotly_3D_cams(my_lsd, my_title, res_dir, frames_idx=frames_idx, save=save, plot=plot)
    my_plot.plotly_3D_cams(comp_lsd, comp_title, res_dir, frames_idx=frames_idx, save=save, plot=plot)

     # write stats
    # TODO uncomment
    stats.append(f'elapsed: {elapsed_time:.0f} sec, avg_per_frame={elapsed_time/endframe:.1f} sec')
    stats += rot_trans_stats
    stats.append('*************\n')
    print("\n".join(stats))
    with open (os.path.join(res_dir, 'stats.txt'), 'a+') as f:
        f.writelines('\n'.join(stats))


    return rots_total_error, trans_total_error

