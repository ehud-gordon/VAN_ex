from itertools import compress

import numpy as np
import matplotlib.pyplot as plt

import os
import time
from pathlib import Path
import shutil

import kitti, features, triang, tracks, utils
import my_plot
from pnp import PNP


class Drive:
    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.k, self.ext_l0_l1, self.ext_li_ri = kitti.read_cameras(dataset_path=self.dataset_path) # k=(3,4) ext_l0/r0 (4,4)
        self.k3 = self.k[:,:3] # (3,3)
        args.out_path = os.path.join(args.out_dir, utils.get_time_path())
        args.out_path += ('_kitti' if args.kitti else '_mine') + ( '_global' if args.globaly else '_relative') + f'_{args.endframe}'
        self.args = args


    def main(self):
        args = self.args
        # match l0-r0, l1-r1
        pnp = PNP(k=self.k, ext_li_ri=self.ext_li_ri)
        featurez = features.Features(args=args)
        # match l0-r0, l1-r1
        start = time.time()
        kp_l0, desc_l0, kp_r0 = featurez.get_kps_desc_stereo_pair(idx=0)
        tracks_db = tracks.Tracks_DB(args=args)
        ext_l0_l1 = self.ext_l0_l1
        ext_l_prev = self.ext_l0_l1
        dws_global = [np.array([0,0,0])]
        r0_to_r1_s_mine, t0_to_t1_s_mine  = [], []
        kp_l1_inlier_matches = []
        global_ext_l0_li = self.ext_l0_l1 # (4,4)
        pc_l0_r0 = triang.triang(kpA=kp_l0, kpB=kp_r0, k=self.k, mA=self.ext_l0_l1, mB=self.ext_li_ri)  # (3,n)
        kp_l0, desc_l0, kp_r0, pc_l0_r0 = triang.filter_based_on_triang(kp_l=kp_l0, desc_l=desc_l0,kp_r=kp_r0, pc=pc_l0_r0)
        l1_bad_inds = None
        for l1_idx in range(1, args.endframe+1): # range(1,2761)
            kp_l1, desc_l1, kp_r1 = featurez.get_kps_desc_stereo_pair(idx=l1_idx)
            pc_l1_r1_temp = triang.triang(kpA=kp_l1, kpB=kp_r1, k=self.k, mA=self.ext_l0_l1, mB=self.ext_li_ri)  # (3,n)
            kp_l1, desc_l1, kp_r1, pc_l1_r1_temp = triang.filter_based_on_triang(kp_l=kp_l1, desc_l=desc_l1, kp_r=kp_r1, pc=pc_l1_r1_temp)
            # match l0-l1
            matches_l0_l1 = featurez.matcher.match(queryDescriptors=desc_l0, trainDescriptors=desc_l1)  # list of matches [DMatch1,... DMatch1N]

            # I have found that l1_bad_inds doesn't improve
            matches_l0_l1 = features.filter_matches(matches_l0_l1, kp0=kp_l0, kp1=kp_l1, stereo_filter=False)
            # matches_l0_l1 = features.filter_matches(matches_l0_l1, kp0=kp_l0, kp1=kp_l1, stereo_filter=False,l1_bad_inds=l1_bad_inds)

            # get ext_l0_l1
            if args.kitti:
                ext_l0_l1 = kitti.read_poses_world_to_cam([l1_idx])[0]  # world_left_0 to world_left_i (camera)
                ext_inliers_bool = np.ones(len(matches_l0_l1))
            else:
                pnp.set_with_matches(matches_l0_l1=matches_l0_l1, kp_l0=kp_l0, kp_l1=kp_l1, pc_l0_r0=pc_l0_r0, kp_r1=kp_r1)
                ext_l0_l1, ext_inliers_bool, proj_errors_to_l1 = pnp.pnp_ransac()  # (4,4), (len(matches)
                l1_matched_inds = [m.trainIdx for m in matches_l0_l1] # 388
                l1_bad_inds = set(compress(l1_matched_inds, ~ext_inliers_bool))

            # compute pc_l1_r1
            if args.globaly:
                pc_l1_r1 = triang.triang(kpA=kp_l1, kpB=kp_r1, k=self.k, mA=ext_l0_l1, mB=(self.ext_li_ri @ ext_l0_l1))  # (3,n)
                global_ext_l0_li = ext_l0_l1

            else: # relative
                pc_l1_r1 = triang.triang(kpA=kp_l1, kpB=kp_r1, k=self.k, mA=self.ext_l0_l1, mB=self.ext_li_ri)  # (3,n)
                global_ext_l0_li = ext_l0_l1 @ global_ext_l0_li

            consistent_matches_l0_l1 = list(compress(matches_l0_l1, ext_inliers_bool))
            if args.store_tracks:
                tracks_db.add_frame(matches_l0_l1=consistent_matches_l0_l1, l1_id=l1_idx, kp_l0=kp_l0, kp_r0=kp_r0, kp_l1=kp_l1, kp_r1=kp_r1,
                                    ext_l1=ext_l0_l1, pc_l0_r0=pc_l0_r0, pc_l1_r1=pc_l1_r1)

            # eval performance #####
            if args.plot or args.save:
                kp_l1_inlier_matches.append((kp_l1.shape[1], sum(ext_inliers_bool), len(matches_l0_l1)))
                # l1 camera place in l0 coordinates
                l1_dw_global = utils.get_dw_from_extrinsics(global_ext_l0_li) # (3,1) local
                dws_global.append(l1_dw_global)
                # relative performance
                r0_to_r1_mine, t0_to_t1_mine = utils.r0_to_r1_t0_to_t1(l0=ext_l_prev, l1=global_ext_l0_li)
                r0_to_r1_s_mine.append(r0_to_r1_mine)
                t0_to_t1_s_mine.append(t0_to_t1_mine)


            ext_l_prev = ext_l0_l1
            kp_l0 = kp_l1
            kp_r0 = kp_r1
            desc_l0 = desc_l1
            pc_l0_r0 = pc_l1_r1
            if l1_idx % 20 == 0:
                print(f'finished frame {l1_idx}')

        ############################################ PLOTS ############################################
        elapsed_time = time.time() - start; avg_time = elapsed_time / args.endframe
        if not (args.save or args.plot or args.store_tracks):
            return

        utils.make_dir(args.out_path)


        r0_to_r1_s_kitti, t0_to_t1_s_kitti = kitti.read_relative_poses_world_to_cam(idx=list(range(0,args.endframe+1)))
        t0_to_t1_s_mine = np.array(t0_to_t1_s_mine).T
        rot_diffs_relative = np.array([utils.rotation_matrices_diff(r, q) for r,q in zip (r0_to_r1_s_mine, r0_to_r1_s_kitti)])
        trans_diffs_relative = np.abs(t0_to_t1_s_kitti - t0_to_t1_s_mine)
        rot_trans_stats, rots_total_error,  trans_total_error = utils.rot_trans_stats(rot_diffs_relative,trans_diffs_relative, endframe=args.endframe)

        stats = ["**Stage2**", str(args), f'elapsed: {elapsed_time:.0f} sec, avg_per_frame={avg_time:.1f} sec'] + rot_trans_stats
        print("\n".join(stats))

        stage2_dir = os.path.join(args.out_path, 'stage2' + f'_{trans_total_error:.1f}_{rots_total_error:.1f}')
        utils.make_dir(stage2_dir)
        if args.store_tracks:
            tracks_db.args = args
            tracks_db.serialize(dir_path=stage2_dir)

        if args.save:
            with open (os.path.join(stage2_dir,'stats.txt'), 'w') as f:
                f.writelines('\n'.join(stats))

        my_plot.plt_diff_rot_matrices(rot_diffs_relative,stage2_dir, "stage2", plot=args.plot, save=args.save)
        my_plot.plt_diff_trans_vecs(trans_diffs_relative, stage2_dir, title="stage2", plot=args.plot, save=args.save)
        my_plot.plt_kp_inlier_matches(kp_l1_inlier_matches,stage2_dir, plot=args.plot, save=args.save)

        # plot 2D/3D left_i camera location comparison
        dws_global = np.array(dws_global).T  # (3, seq_leqngth-1)
        dws_kitti = kitti.read_dws(list(range(args.endframe + 1)))
        my_plot.plotly_3d_cams_compare(dws_global, dws_kitti, stage2_dir, title="stage2", endframe=args.endframe, save=args.save)
        my_plot.plt_2d_cams_compare(my_dws=dws_global,kitti_dws=dws_kitti, plot_dir=stage2_dir, title="stage2", plot=args.plot, save=args.save)
        my_plot.plt_3d_cams_compare(my_dws=dws_global,kitti_dws=dws_kitti, plot_dir=stage2_dir, title="stage2", plot=args.plot, save=args.save)