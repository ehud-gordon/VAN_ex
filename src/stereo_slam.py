import numpy as np

import os
import time
from itertools import compress

from utils import utils_img
import kitti, features, tracks, utils
import results
from plot import my_plot
from calib3d import pnp, triang


class Drive:

    def __init__(self, args):
        self.args = args
        self.k, self.ext_id, self.ext_l_to_r = kitti.read_cameras(args.dataset_path) # k=(3,4) ext_l0/r0 (4,4)

    def main(self):
        # init
        args = self.args
        featurez = features.Features(**vars(args))
        ext_li_to_lj_s = [np.diag([1,1,1,1])]
        tracks_db = tracks.Tracks_DB()
        kp_lj_inlier_matches = []
        
        # match li-ri
        start_time = time.time()
        kp_li, desc_li, kp_ri = featurez.get_kps_desc_stereo_pair(idx=args.startframe)
        kp_li, kp_ri, pc_lr_i_in_li, desc_li = triang.triang_and_rel_filter(kp_li, kp_ri, self.k, self.ext_id, self.ext_l_to_r, desc_li)

        for j in args.frames_idx[1:]: # [1,2,...,2760]
            i=j-1
            # match lj-rj
            kp_lj, desc_lj, kp_rj = featurez.get_kps_desc_stereo_pair(j)
            kp_lj, kp_rj, pc_lr_j_in_lj, desc_lj = triang.triang_and_rel_filter(kp_lj, kp_rj, self.k, self.ext_id, self.ext_l_to_r, desc_lj)  # (3,n)

            # match li-lj
            matches_li_lj = featurez.matcher.match(desc_li.T, desc_lj.T)  # list of matches [DMatch1,... DMatch1N]
            matches_li_lj = features.filter_matches(matches_li_lj, kp_li, kp_lj, is_stereo=False)

            # get ext_l0_lj with pnp
            pc_i_in_li_matched, kp_lj_matched, kp_rj_matched = features.filter_with_matches(matches_li_lj, [pc_lr_i_in_li], [kp_lj, kp_rj])
            ext_li_to_lj, ext_inliers_bool, proj_errors_lj, proj_errors_rj = pnp.pnp_ransac(kp_lj_matched, kp_rj_matched,
                                                                                            pc_i_in_li_matched, self.k, self.ext_l_to_r, j)  # (4,4), (len(matches))
            ext_li_to_lj_s.append(ext_li_to_lj)
            pnp_inliers_matches_li_lj = list(compress(matches_li_lj, ext_inliers_bool))
            
            if args.store_tracks:
                tracks_db.add_frame(pnp_inliers_matches_li_lj, i,j,
                                    kp_li, kp_ri, kp_lj, kp_rj,
                                    pc_lr_i_in_li, pc_lr_j_in_lj)

            # used for visualization
            kp_lj_inlier_matches.append( (kp_lj.shape[1], sum(ext_inliers_bool), len(matches_li_lj)) )

            kp_li = kp_lj
            kp_ri = kp_rj
            desc_li = desc_lj
            pc_lr_i_in_li = pc_lr_j_in_lj
            if (j-args.startframe) % 20 == 0:
                print(f'finished frame {j}')
        
        self.output_results(kp_lj_inlier_matches, ext_li_to_lj_s, tracks_db, start_time)
        return ext_li_to_lj_s, tracks_db
    
    def output_results(self, kp_lj_inlier_matches, ext_li_to_lj_s, tracks_db, start_time):
        args = self.args
        stats = ['**STAGE2**', str(args)]
        # serialize tracks and ext
        if args.store_tracks:
            self.stage2_tracks_path = tracks_db.serialize(args.stage2_dir, f'stage2_tracks_{args.endframe}')
        utils.serialize_ext_li_to_lj_s(args.stage2_dir, ext_li_to_lj_s, title=f'stage2_{args.endframe}')

        ext_li_to_l0_s = utils_img.concat_and_inv_ci_to_cj_s(ext_li_to_lj_s)
        # output important plots and stats
        rots_total_error, trans_total_error = results.output_results(args.out_path, ext_li_to_l0_s, args.frames_idx, "stage_2",
                                                                     start_time, plot=args.plot, save=args.save)
        
        # create folder
        args.stage2_dir =  os.path.join(args.out_path, 'stage2' + f'_{trans_total_error:.1f}_{rots_total_error:.1f}')
        os.makedirs(args.stage2_dir)

        
        # output keypoints, matches, inliers counts
        kp_lj_inlier_matches = np.array(kp_lj_inlier_matches).T
        avg_num_of_keypoints = np.mean(kp_lj_inlier_matches[0])
        matches_frac_out_of_keypoints = np.mean( kp_lj_inlier_matches[2] / kp_lj_inlier_matches[0])
        inliers_frac_out_of_matches = np.mean(kp_lj_inlier_matches[1] / kp_lj_inlier_matches[2])
        tmp_d = {'kp_lj':kp_lj_inlier_matches[0], 'li_lj_matches':kp_lj_inlier_matches[2], 'pnp_inliers':kp_lj_inlier_matches[1]}
        tmp_title = f"count_kp_left1_matches_inliers_{args.startframe}_{args.endframe}"
        my_plot.plotly_scatters(tmp_d, x=args.frames_idx[1:], title=tmp_title, plot_dir=args.stage2_dir, plot=args.plot, save=args.save, yaxis="count")
        stats.append(f'avg. number of detected keypoints: {avg_num_of_keypoints:.0f}', 
                     f'avg. percent of matched out of keypoints {matches_frac_out_of_keypoints:.0%}',
                     f'avg. percent of inliers out of matches {inliers_frac_out_of_matches:.0%}'
                     )

        # write stats
        with open (os.path.join(args.stage2_dir ,'stats_stage2.txt'), 'w') as f:
            f.writelines('\n'.join(stats))