import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

import my_code.utils as utils
import kitti, features, triang, tracks
from pnp import PNP



class Drive:
    def __init__(self, dataset_path=kitti.DATASET_5_PATH):
        self.dataset_path = dataset_path
        self.k, self.ext_l0, self.ext_r0 = kitti.read_cameras(dataset_path) # k=(3,4) ext_l0/ext_r0 (4,4)
        self.k3 = self.k[:,:3] # (3,3)

    def main(self):
        keep_trying = True
        # match l0-r0, l1-r1
        pnp = PNP(k=self.k, ext_r0=self.ext_r0)
        seq_length = kitti.get_seq_length(dataset_path=self.dataset_path) # 2760
        seq_length = 3
        save = False; plot = False

        while keep_trying:
            # match l0-r0, l1-r1
            kp_l0, desc_l0, kp_r0 = features.get_kps_desc_stereo_pair(idx=0)
            tracks_db = tracks.Tracks(kp_l0=kp_l0, kp_r0=kp_r0)
            ext_l0 = self.ext_l0
            l1_dws = []; l1_dws_pose = []
            left1_ext_dists = []
            tx_error, ty_error, tz_error = [], [], []
            start = time.time()
            kp_l1_inlier_matches = []
            # global_ext_l_i = self.ext_l0 # (4,4) # local


            keep_trying = False
            for l1_idx in range(1,seq_length+1):
                pc_l0_r0 = triang.triang(kp1=kp_l0, kp2=kp_r0, k=self.k, m1=ext_l0, m2=(self.ext_r0 @ ext_l0)) # (3,n) # global
                # pc_l0_r0 = triang.triang(kp1=kp_l0, kp2=kp_r0, k=self.k, m1=self.ext_l0, m2=self.ext_r0)  # (3,n) # local
                kp_l1, desc_l1, kp_r1 = features.get_kps_desc_stereo_pair(idx=l1_idx)

                # match l0-l1
                knn_matches_l0_l1 = features.match_desc_knn(desc1=desc_l0, desc2=desc_l1)
                # query is 0 side, train is 1 side. The holy grail
                matches_l0_l1 = features.filter_knn_matches(knn_matches_l0_l1, kp1=kp_l0, kp2=kp_l1, stereo_filter=False)
                # filt_l1_l0 = {m.trainIdx: m.queryIdx for m in matches_l0_l1}
                tracks_db.add_frame(matches_l0_l1=matches_l0_l1, l1_idx=l1_idx, kp_l1=kp_l1, kp_r1=kp_r1)
                # get ext_l1 from RANSAC
                pnp.set_with_matches(matches_l0_l1=matches_l0_l1, kp_l0=kp_l0, kp_l1=kp_l1, pc_l0_r0=pc_l0_r0, kp_r1=kp_r1)
                ext_l1, ext_l1_bool = pnp.pnp_ransac(iters=100)  # (4,4)
                # global_ext_l_i = ext_l1 @ global_ext_l_i # local

                kp_l1_inlier_matches.append((kp_l1.shape[1], sum(ext_l1_bool), len(matches_l0_l1)))
                # l1 camera place in l0 coordinates
                l1_dw = utils.get_dw_from_extrinsics(ext_l1) # (3,1) global
                # l1_dw = utils.get_dw_from_extrinsics(global_ext_l_i) # (3,1) local
                l1_dws.append(l1_dw)
                # pose
                ext_l1_pose = kitti.read_poses_world_to_cam([l1_idx])[0]  # FROM WORLD (l0) TO CAMERA (l1), (4,4)

                l1_dw_pose = utils.get_dw_from_extrinsics(ext_l1_pose) # (3,1)
                l1_dws_pose.append(l1_dw_pose)

                tx_error.append(abs(ext_l1_pose[0, 3] - ext_l1[0, 3])); ty_error.append(abs(ext_l1_pose[1, 3] - ext_l1[1, 3])); tz_error.append(abs(ext_l1_pose[2, 3] - ext_l1[2, 3]))  # global
                # tx_error.append(abs(ext_l1_pose[0, 3] - global_ext_l_i[0, 3])); ty_error.append(abs(ext_l1_pose[1, 3] - global_ext_l_i[1, 3]));tz_error.append(abs(ext_l1_pose[2, 3] - global_ext_l_i[2, 3])) # local
                left1_ext_dist = np.sum(abs(ext_l1_pose-ext_l1)); left1_ext_dists.append(left1_ext_dist) # global
                # left1_ext_dists = np.sum(abs(ext_l1_pose - global_ext_l_i)); left1_ext_dists.append(left1_ext_dist) # local
                print(f"{l1_idx}: {kp_l1_inlier_matches[-1]}, left1_ext_dist:{left1_ext_dist:.2f}")

                if left1_ext_dist > 700:
                    keep_trying = True
                    print(f"l1_idx={l1_idx}, mat dist too large: {left1_ext_dist:.0f}")
                    break

                ext_l0 = ext_l1
                kp_l0 = kp_l1
                kp_r0 = kp_r1
                desc_l0 = desc_l1
                if l1_idx % 20 == 0:
                    print(f'finished frame {l1_idx}')

        tracks_db.serialize()
        elapsed_time = time.time()-start; avg_time = elapsed_time / seq_length
        print(f'elapsed: {elapsed_time:.0f} sec, avg_per_frame={avg_time:.1f} sec')
        print(f"avg_left1_ext_dist={sum(left1_ext_dists)/len(left1_ext_dists):.0f} when creating pc with triangulation")
        print(f"tx_error:{sum(tx_error):.0f}"); print(f"ty_error:{sum(ty_error):.0f}"); print(f"tz_error:{sum(tz_error):.0f}")
        # for l1_dw, l1_dw_pose in zip(l1_dws, l1_dws_pose):
        #     print(l1_dw); print(l1_dw_pose); print()
        if not plot:
            return
        l1_dws = np.array(l1_dws).T # (3, seq_length-1)
        l1_dws_pose = np.array(l1_dws_pose).T  # (3,seq_length-1)

        # plot left1_ext_dists per frame
        rng = np.arange(seq_length)+1 # [1-2760]
        plt.figure(); plt.plot(rng, left1_ext_dists); plt.xlabel('frame'); plt.ylabel('L1 Norm btwn extrinsics')
        plt.title("L1 Norm btwn my global left_i extrinsics and pose left_i extrinsics")
        if save:
            path = os.path.join(utils.FIG_PATH, f'left1_ext_dist_{seq_length}' + '.png'); path = utils.get_avail_path(path); plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.show()

        # plot tx, ty, tz L1 distance
        plt.figure(); plt.plot(rng, tx_error, label="tx"); plt.plot(rng, ty_error, label="ty"); plt.plot(rng, tz_error, label="tz")
        plt.legend(); plt.xlabel('frame'); plt.ylabel('L1 Norm')
        plt.title("L1 Norm btwn translation vector of my left_i extrinsic and pose left_i extrinsic")
        path = os.path.join(utils.FIG_PATH, f'trans_vectors_dist_{seq_length}' + '.png'); path = utils.get_avail_path(path); plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.show()

        # plot number of kp, inlier and matches
        fig = plt.figure()
        plt.plot(rng, [t[0] for t in kp_l1_inlier_matches], label="kp_left0")
        plt.plot(rng, [t[1] for t in kp_l1_inlier_matches], label="inlier");plt.plot(rng, [t[2] for t in kp_l1_inlier_matches], label="matches")
        plt.xlabel('frame'); plt.ylabel('count'); plt.legend(); plt.show()

        # plot 2d left_i camera location comparison
        fig = plt.figure()
        plt.scatter(x=l1_dws[2], y=l1_dws[0], color="blue", label="mine")
        plt.scatter(x=l1_dws_pose[2], y=l1_dws_pose[0], color="red", label="pose",alpha=0.4)
        plt.gca().invert_yaxis()
        plt.xlabel('z');plt.ylabel('x')
        plt.legend()
        plt.title(f"comparing left_i camera location for {seq_length} frames")
        path = os.path.join(utils.FIG_PATH, f'l1_path_above_{seq_length}' + '.png');path = utils.get_avail_path(path)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.show()

        # plot 3D left1_camera comparison
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(l1_dws[0], l1_dws[2], l1_dws[1], color="blue", label="mine")
        ax.scatter(l1_dws_pose[0], l1_dws_pose[2], l1_dws_pose[1], color="red", label="pose", alpha=0.4)
        xmin, ymin, zmin = np.min(l1_dws, axis=1)
        xmax, ymax, zmax = np.max(l1_dws, axis=1)
        ax.set_ylim([0, zmax + 1])  # not a mistake, plt's Y axis is our Z-Axis
        ax.set_xlim([xmin - 1, xmax + 1])
        ax.set_zlim([ymin - 1, ymax + 1])  # not a mistake, plt's z-axis is our Y-axis
        ax.invert_zaxis()  # not a mistake, - plt's z axis is our Y axis
        ax.set_xlabel('X');ax.set_ylabel('Z'); ax.set_zlabel('Y')  # not a mistake
        plt.legend()
        plt.title(f"comparing left_i camera location for {seq_length} frames")
        path = os.path.join(utils.FIG_PATH, f'l1_path_3d_{seq_length}' + '.png'); path = utils.get_avail_path(path)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.show()

        x=3
