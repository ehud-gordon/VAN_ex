import numpy as np
import matplotlib.pyplot as plt

import os
import time
from pathlib import Path
import shutil

import kitti, features, triang, tracks, utils
from pnp import PNP

class Drive:
    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.k, self.ext_l0, self.ext_r0 = kitti.read_cameras(dataset_path=self.dataset_path) # k=(3,4) ext_l0/r0 (4,4)
        self.k3 = self.k[:,:3] # (3,3)
        self.args = args

    def main(self):
        args = self.args
        # match l0-r0, l1-r1
        pnp = PNP(k=self.k, ext_r0=self.ext_r0)
        featurez = features.Features(args=args)
        # match l0-r0, l1-r1
        start = time.time()
        kp_l0, desc_l0, kp_r0 = featurez.get_kps_desc_stereo_pair(idx=0)
        tracks_db = tracks.Tracks_DB(args=args)
        ext_l_prev = self.ext_l0
        dws_global = [np.array([0,0,0])]
        trans_diffs_relative, rot_diffs_relative = [], []
        kp_l1_inlier_matches = []
        global_ext_l_i = self.ext_l0 # (4,4)
        pc_l0_r0 = triang.triang(kp1=kp_l0, kp2=kp_r0, k=self.k, m1=self.ext_l0, m2=self.ext_r0)  # (3,n)
        kp_l0, desc_l0, kp_r0, pc_l0_r0 = triang.filter_based_on_triang(kp_l=kp_l0, desc_l=desc_l0,kp_r=kp_r0, pc=pc_l0_r0)

        for l1_idx in range(1, args.endframe+1): # range(1,2761)
            kp_l1, desc_l1, kp_r1 = featurez.get_kps_desc_stereo_pair(idx=l1_idx)
            pc_l1_r1_temp = triang.triang(kp1=kp_l1, kp2=kp_r1, k=self.k, m1=self.ext_l0, m2=self.ext_r0)  # (3,n)
            kp_l1, desc_l1, kp_r1, pc_l1_r1_temp = triang.filter_based_on_triang(kp_l=kp_l1, desc_l=desc_l1, kp_r=kp_r1, pc=pc_l1_r1_temp)
            # match l0-l1
            knn_matches_l0_l1 = featurez.match_desc_knn(desc1=desc_l0, desc2=desc_l1)
            # query is 0 side, train is 1 side. The holy grail
            matches_l0_l1 = features.filter_knn_matches(knn_matches_l0_l1, kp1=kp_l0, kp2=kp_l1, stereo_filter=False)

            # get ext_l1
            if args.kitti:
                ext_l1 = kitti.read_poses_world_to_cam([l1_idx])[0]  # l0 to li
                ext_l1_bool = np.ones(len(matches_l0_l1))
            else:
                pnp.set_with_matches(matches_l0_l1=matches_l0_l1, kp_l0=kp_l0, kp_l1=kp_l1, pc_l0_r0=pc_l0_r0, kp_r1=kp_r1)
                ext_l1, ext_l1_bool = pnp.pnp_ransac()  # (4,4), (len(matches),

            if args.globaly:
                pc_l1_r1 = triang.triang(kp1=kp_l1, kp2=kp_r1, k=self.k, m1=ext_l1, m2=(self.ext_r0 @ ext_l1))  # (3,n)
                global_ext_l_i = ext_l1
            else: # relative
                pc_l1_r1 = triang.triang(kp1=kp_l1, kp2=kp_r1, k=self.k, m1=self.ext_l0, m2=self.ext_r0)  # (3,n)
                global_ext_l_i = ext_l1 @ global_ext_l_i


            if args.store_tracks:
                tracks_db.add_frame(matches_l0_l1=matches_l0_l1, l1_id=l1_idx, kp_l0=kp_l0, kp_r0=kp_r0, kp_l1=kp_l1, kp_r1=kp_r1,
                                    ext_l1=ext_l1, pc_l0_r0=pc_l0_r0, pc_l1_r1=pc_l1_r1)

            # eval performance #####
            if args.plot or args.save:
                kp_l1_inlier_matches.append((kp_l1.shape[1], sum(ext_l1_bool), len(matches_l0_l1)))
                # l1 camera place in l0 coordinates
                l1_dw_global = utils.get_dw_from_extrinsics(global_ext_l_i) # (3,1) local
                dws_global.append(l1_dw_global)
                # relative performance
                r0_to_r1_kitti, t0_to_t1_kitti = kitti.read_relative_poses_world_to_cam(idx=[l1_idx])
                r0_to_r1_mine, t0_to_t1_mine = utils.get_l0_to_l1_trans_rot(l0=ext_l_prev, l1=global_ext_l_i)
                trans_diffs_relative.append(abs(t0_to_t1_kitti - t0_to_t1_mine))
                rot_diffs_relative.append(utils.rotation_matrices_diff(r0_to_r1_kitti, r0_to_r1_mine))


            ext_l_prev = ext_l1
            kp_l0 = kp_l1
            kp_r0 = kp_r1
            desc_l0 = desc_l1
            pc_l0_r0 = pc_l1_r1
            if l1_idx % 20 == 0:
                print(f'finished frame {l1_idx}')

        time_path = utils.get_time_path()
        if args.store_tracks:
            title = time_path + ('_kitti' if args.kitti else '_mine') + ('_global' if args.globaly else '')
            tracks_db.serialize(title=title)


        ############################################ PLOTS ############################################
        if not args.plot:
            return
        trans_diffs_relative = np.array(trans_diffs_relative).T  # (3,10)
        tx_error, ty_error, tz_error = np.sum(trans_diffs_relative, axis=1)
        trans_total_error = + tx_error + ty_error + tz_error
        trans_avg_error = trans_total_error / args.endframe
        rots_total_error = np.sum(rot_diffs_relative)
        rot_avg_error = rots_total_error / args.endframe
        elapsed_time = time.time() - start; avg_time = elapsed_time / args.endframe
        stats = [str(args),
                 f'elapsed: {elapsed_time:.0f} sec, avg_per_frame={avg_time:.1f} sec',
                 f"avg. relative translation error = {trans_avg_error:.1f}, total translation error = {trans_total_error:.1f}",
                 f"avg. relative rotation error = {rot_avg_error:.2f} deg, total rotation error = {rots_total_error:.1f} deg",
                 f"tx_error:{tx_error:.1f}", f"ty_error:{ty_error:.1f}", f"tz_error:{tz_error:.1f}"]
        print("\n".join(stats))
        # for l1_dw, l1_dw_kitt in zip(l1_dws, l1_dws_kitti):
        #     print(l1_dw); print(l1_dw_kitti); print()

        plot_dir = ""
        if args.save:
            plot_dir = os.path.join(utils.fig_path(), time_path + f'_{trans_total_error:.1f}_{rots_total_error:.1f}_{args.endframe}')
            utils.make_dir(plot_dir)
            with open (os.path.join(plot_dir,'stats.txt'), 'w') as f:
                f.writelines('\n'.join(stats))
        dws_global = np.array(dws_global).T # (3, seq_leqngth-1)
        dws_kitti = kitti.read_dws(list(range(args.endframe+1)))

        rng = np.arange(args.endframe)+1  # [1-2760] inclusive
        # plot tx, ty, tz, rots L1 relative difference
        plt.figure(); plt.plot(rng, trans_diffs_relative[0], label="tx"); plt.plot(rng, trans_diffs_relative[1], label="ty"); plt.plot(rng, trans_diffs_relative[2], label="tz")
        plt.legend(); plt.xlabel('frame'); plt.ylabel('L1 Norm')
        plt.title("L1 Norm relative, btwn my and kitti's left1 extrinsic matrices")
        if args.save:
            path = os.path.join(plot_dir, f'trans_relative_diff_{args.endframe}' + '.png')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.show()

        # plot rot diff
        plt.figure(); plt.plot(rng, rot_diffs_relative, label="rot")
        plt.legend(); plt.xlabel('frame'); plt.ylabel('degrees')
        plt.title("difference in degrees, relative,btwn my and kitti's left1 extrinsic matrices")
        if args.save:
            path = os.path.join(plot_dir, f'rot_relative_diff_{args.endframe}' + '.png')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.show()

        # plot number of kp, inlier and matches
        plt.figure()
        plt.plot(rng, [t[0] for t in kp_l1_inlier_matches], label="keypoints_left1")
        plt.plot(rng, [t[1] for t in kp_l1_inlier_matches], label="inliers");plt.plot(rng, [t[2] for t in kp_l1_inlier_matches], label="matches")
        plt.xlabel('frame'); plt.ylabel('count')
        plt.title('count of inliers / matches / (keypoints in left1)')
        if args.save:
            path = os.path.join(plot_dir, f'kp_l1_inlier_matches_{args.endframe}' + '.png')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.legend(); plt.show()

        # plot 2D left_i camera location comparison
        plt.figure()
        plt.scatter(x=dws_global[0], y=dws_global[2], color="blue", label="mine")
        plt.scatter(x=dws_kitti[0], y=dws_kitti[2], color="red", label="kitti",alpha=0.4)
        plt.xlabel('x');plt.ylabel('z')
        plt.legend()
        plt.title(f"Comparing left camera location in frames [1,{args.endframe}]")
        if args.save:
            path = os.path.join(plot_dir, f'l1_path_above_{args.endframe}' + '.png')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.show()


        # plot 3D left1_camera comparison
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(dws_global[0], dws_global[2], dws_global[1], color="blue", label="mine")
        ax.scatter(dws_kitti[0], dws_kitti[2], dws_kitti[1], color="red", label="kitti", alpha=0.4)
        xmin, ymin, zmin = np.min(dws_global, axis=1)
        xmax, ymax, zmax = np.max(dws_global, axis=1)
        ax.set_ylim([0, zmax + 1])  # not a mistake, plt's Y axis is our Z-Axis
        ax.set_xlim([xmin - 1, xmax + 1])
        ax.set_zlim([ymin - 1, ymax + 1])  # not a mistake, plt's z-axis is our Y-axis
        ax.invert_zaxis()  # not a mistake, - plt's z axis is our Y axis
        ax.set_xlabel('X');ax.set_ylabel('Z'); ax.set_zlabel('Y')  # not a mistake
        plt.legend()
        plt.title(f"Comparing left camera location in frames [1,{args.endframe}]")
        if args.save:
            path = os.path.join(plot_dir, f'l1_path_3d_{args.endframe}' + '.png')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.show()