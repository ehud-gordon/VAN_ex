import numpy as np

import os
import time
from itertools import compress

import kitti, features, triang, tracks, utils
import my_plot, results
from pnp import PNP

class Drive:
    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.k, self.init_ext_l0_l1, self.ext_li_ri = kitti.read_cameras(self.dataset_path) # k=(3,4) ext_l0/r0 (4,4)
        self.k3 = self.k[:,:3] # (3,3)
        args.out_path = os.path.join(args.out_dir, utils.get_time_path())
        args.out_path += ('_kitti' if args.kitti else '_mine') + ( '_relative' if args.relative else '_global') + f'_{args.endframe}'
        self.args = args

        if args.store_tracks or args.save or args.plot:
            utils.clear_and_make_dir(args.out_path)


    def main(self):
        # init
        args = self.args
        pnp = PNP(k=self.k, ext_li_ri=self.ext_li_ri)
        featurez = features.Features(args=args)
        ext_l0_to_l1_s = [np.diag((1,1,1,1))]
        self.tracks_db = tracks.Tracks_DB(args=args)
        kp_l1_inlier_matches = []
        
        # match l0-r0, l1-r1
        start_time = time.time()
        kp_l0, desc_l0, kp_r0 = featurez.get_kps_desc_stereo_pair(idx=0)
        kp_l0, desc_l0, kp_r0, pc_l0_r0 = triang.triang_and_filter(kp_l0, kp_r0, self.k, self.init_ext_l0_l1, self.ext_li_ri, desc_l0)
        
        for l1_idx in range(1, args.endframe+1): # range(1,2761)
            kp_l1, desc_l1, kp_r1 = featurez.get_kps_desc_stereo_pair(l1_idx)
            kp_l1, desc_l1, kp_r1, _ = triang.triang_and_filter(kp_l1, kp_r1, self.k, self.init_ext_l0_l1, self.ext_li_ri, desc_l1)  # (3,n)

            # match l0-l1
            matches_l0_l1 = featurez.matcher.match(desc_l0, desc_l1)  # list of matches [DMatch1,... DMatch1N]
            matches_l0_l1 = features.filter_matches(matches_l0_l1, kp_l0, kp_l1, is_stereo=False)

            # get ext_l0_l1
            if args.kitti:
                ext_l0_l1 = kitti.read_poses_world_to_cam([l1_idx])[0]  # world_left_0 to world_left_i (camera)
                ext_inliers_bool = np.ones(len(matches_l0_l1))
            else:
                pnp.set_with_matches(matches_l0_l1, kp_l0, kp_l1, pc_l0_r0, kp_r1)
                ext_l0_l1, ext_inliers_bool, proj_errors_to_l1 = pnp.pnp_ransac()  # (4,4), (len(matches)

            ext_l0_to_l1_s.append(ext_l0_l1)
            # compute pc_l1_r1
            if args.relative:
                pc_l1_r1 = triang.triang(kp_l1, kp_r1, self.k, self.init_ext_l0_l1, self.ext_li_ri)  # (3,n)
            else: # global
                pc_l1_r1 = triang.triang(kp_l1, kp_r1, self.k, ext_l0_l1, self.ext_li_ri @ ext_l0_l1)  # (3,n)
                

            consistent_matches_l0_l1 = list(compress(matches_l0_l1, ext_inliers_bool))
            if args.store_tracks:
                self.tracks_db.add_frame(matches_l0_l1=consistent_matches_l0_l1, l1_id=l1_idx,
                                    kp_l0=kp_l0, kp_r0=kp_r0, kp_l1=kp_l1, kp_r1=kp_r1,
                                    pc_l0_r0=pc_l0_r0, pc_l1_r1=pc_l1_r1)

            # used for visualization
            kp_l1_inlier_matches.append((kp_l1.shape[1], sum(ext_inliers_bool), len(matches_l0_l1)))

            kp_l0 = kp_l1
            kp_r0 = kp_r1
            desc_l0 = desc_l1
            pc_l0_r0 = pc_l1_r1
            if l1_idx % 20 == 0:
                print(f'finished frame {l1_idx}')
        
        self.tracks_db.ext_l1s = ext_l0_to_l1_s
        if args.save or args.plot or args.store_tracks:
            self.output_results(kp_l1_inlier_matches, ext_l0_to_l1_s, self.tracks_db, start_time)

    
    def output_results(self, kp_l1_inlier_matches, ext_l0_to_l1_s, tracks_db, start_time):
        args = self.args
        frames_idx = list(range(0, args.endframe+1))
        # output important plots and stats
        rots_total_error, trans_total_error = results.output_results(args.out_path, ext_l0_to_l1_s, frames_idx, "stage2",
                                                                     start_time, plot=args.plot, save=args.save)
        
        # create folder
        stage2_dir =  os.path.join(args.out_path, 'stage2' + f'_{trans_total_error:.1f}_{rots_total_error:.1f}')
        os.makedirs(stage2_dir)
        self.tracks_db_path = os.path.join(stage2_dir, f'stage2_tracks_{args.endframe}.pkl')

        # serialzie tracks
        if args.store_tracks:
            tracks_db.serialize(dir_path=stage2_dir)

        # write stats
        stats = ['**STAGE2**', str(args)]
        with open (os.path.join(stage2_dir ,'stats_stage2.txt'), 'w') as f:
            f.writelines('\n'.join(stats))
        
        my_plot.plt_kp_inlier_matches(kp_l1_inlier_matches,stage2_dir, plot=args.plot, save=args.save)