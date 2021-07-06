import numpy as np

import os
import time
from itertools import compress

import kitti, features, triang, tracks, utils
import my_plot, results, pnp

class Drive:

    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.k, self.ext_id, self.ext_l_to_r = kitti.read_cameras(self.dataset_path) # k=(3,4) ext_l0/r0 (4,4)
        self.args = args

    def main(self):
        # init
        args = self.args
        featurez = features.Features(**vars(args))
        ext_l0_to_li_s = [np.diag((1,1,1,1))]
        self.tracks_db = tracks.Tracks_DB()
        kp_lj_inlier_matches = []
        
        # match li-ri
        start_time = time.time()
        kp_li, desc_li, kp_ri = featurez.get_kps_desc_stereo_pair(idx=0)
        kp_li, kp_ri, pc_lr_i_in_l0, desc_li = triang.triang_and_rel_filter(kp_li, kp_ri, self.k, self.ext_id, self.ext_l_to_r, desc_li)

        for j in range(1, args.endframe+1): # [1,2,...,2760]
            i=j-1
            # match lj-rj
            kp_lj, desc_lj, kp_rj = featurez.get_kps_desc_stereo_pair(j)
            kp_lj, kp_rj, pc_lr_j_in_lj, desc_lj = triang.triang_and_rel_filter(kp_lj, kp_rj, self.k, self.ext_id, self.ext_l_to_r, desc_lj)  # (3,n)

            # match li-lj
            matches_li_lj = featurez.matcher.match(desc_li.T, desc_lj.T)  # list of matches [DMatch1,... DMatch1N]
            matches_li_lj = features.filter_matches(matches_li_lj, kp_li, kp_lj, is_stereo=False)

            # get ext_l0_lj with pnp
            pc_i_matched, kp_lj_matched, kp_rj_matched = features.filter_with_matches(matches_li_lj, [pc_lr_i_in_l0], [kp_lj, kp_rj])
            ext_l0_to_lj, ext_inliers_bool, proj_errors_lj, proj_errors_rj = pnp.pnp_ransac(kp_lj_matched, kp_rj_matched,
                                                                                            pc_i_matched, self.k, self.ext_l_to_r,j)  # (4,4), (len(matches))
            ext_l0_to_li_s.append(ext_l0_to_lj)
            pnp_inliers_matches_li_lj = list(compress(matches_li_lj, ext_inliers_bool))

            # TODO below is doing additional rel_then_quant filtering on inliers. send quant_matches instead of pnp_inliers
            # pc_i_pnp_inliers = pc_i_matched[:,ext_inliers_bool]
            # quant_filter = triang.get_quantile_point_cloud_filter(pc_i_pnp_inliers)
            # quant_matches_li_lj = list(compress(pnp_inliers_matches_li_lj, quant_filter))

            # compute pc_lj_rj
            # pc_lr_j_in_l0 below is in WORLD (Left0) CS.
            pc_lr_j_in_l0 = triang.triang(kp_lj, kp_rj, self.k, ext_l0_to_lj, self.ext_l_to_r @ ext_l0_to_lj)  # (3,n)
            # TODO maybe I should futrther filter matches_li_lj using the triang above?
            # TODO make sure it'll work with my tracks matching system!!
            
            if args.store_tracks:
                self.tracks_db.add_frame(pnp_inliers_matches_li_lj, i,j,
                                    kp_li, kp_ri, kp_lj, kp_rj,
                                    pc_lr_i_in_l0, pc_lr_j_in_l0)

            # used for visualization
            kp_lj_inlier_matches.append( (kp_lj.shape[1], sum(ext_inliers_bool), len(matches_li_lj)) )

            kp_li = kp_lj
            kp_ri = kp_rj
            desc_li = desc_lj
            pc_lr_i_in_l0 = pc_lr_j_in_l0
            if j % 20 == 0:
                print(f'finished frame {j}')
        
        self.tracks_db.ext_l0_to_li_s = ext_l0_to_li_s
        
        self.output_results(kp_lj_inlier_matches, ext_l0_to_li_s, self.tracks_db, start_time)

    
    def output_results(self, kp_lj_inlier_matches, ext_l0_to_li_s, tracks_db, start_time):

        args = self.args
        frames_idx = list(range(0, args.endframe+1))
        ext_li_to_l0_s = utils.inv_extrinsics_mult(ext_l0_to_li_s)
        # output important plots and stats
        rots_total_error, trans_total_error = results.output_results(args.out_path, ext_li_to_l0_s, frames_idx, "stage2",
                                                                     start_time, plot=args.plot, save=args.save, relative=args.relative)
        
        # create folder
        args.stage2_dir =  os.path.join(args.out_path, 'stage2' + f'_{trans_total_error:.1f}_{rots_total_error:.1f}')
        os.makedirs(args.stage2_dir)

        # serialize tracks and ext
        if args.store_tracks:
            self.stage2_tracks_path = tracks_db.serialize(dir_path=args.stage2_dir, title=f'stage2_tracks_{args.endframe}')
        utils.serialize_ext_l0_to_li_s(args.stage2_dir, ext_l0_to_li_s, title=f'stage2_{args.endframe}')

        # write stats
        stats = ['**STAGE2**', str(args)]
        with open (os.path.join(args.stage2_dir ,'stats_stage2.txt'), 'w') as f:
            f.writelines('\n'.join(stats))
        
        my_plot.plt_kp_inlier_matches(kp_lj_inlier_matches, args.stage2_dir, plot=args.plot)