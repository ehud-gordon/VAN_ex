import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import os

import kitti
import utils

class PNP:
    def __init__(self, k, ext_li_ri):
        self.k = k # (3,4)
        self.k3 = k[:,:3] # (3,3)
        self.ext_li_ri = ext_li_ri # (4,4) extrinsic, from world_left_i to world_right_i

    def set_with_matches(self, matches_l0_l1, kp_l0, kp_l1, pc_l0_r0, kp_r1):
        """
        :param matches_l0_l1: [DMatch1,...,] between queryIdx in pc_l0 and trainIdx in kp_l1/r1
        :param kp_l0: (2,n) of x-y pixels of kp
        :param pc_l0_r0: point cloud, (3,n) matrix, in world_left_0, filtered by matches from original pc_l0_r0
        :return:
        """
        query_inds = [m.queryIdx for m in matches_l0_l1]
        train_inds = [m.trainIdx for m in matches_l0_l1]

        self.pc_l0_r0 = pc_l0_r0[:,query_inds]
        self.kp_l0 = kp_l0[:,query_inds]
        self.kp_l1 = kp_l1[:,train_inds]
        self.kp_r1 = kp_r1[:,train_inds]

    def set(self, kp_l1, pc_l0_r0, kp_r1):
        """
        :param kp_l1/r1: (2,n) of x-y pixels of kp in l1/r1 that're 4-matched
        :param pc_l0_r0: point cloud, (3,n) matrix, in world_left_0, filtered by matches from original pc_l0_r0
        :return:
        """
        assert kp_l1.shape[1] ==  kp_r1.shape[1] == pc_l0_r0.shape[1]
        self.kp_l1 = kp_l1
        self.pc_l0_r0 = pc_l0_r0
        self.kp_r1 = kp_r1

    def pnp(self):
        pc_l0_4, pxls_l1_4 = get_pc_pxls_for_cv_pnp(self.pc_l0_r0, self.kp_l1, size=4)
        retval, rvec, tvec = cv2.solvePnP(objectPoints=pc_l0_4, imagePoints=pxls_l1_4,
                                                         cameraMatrix=self.k3, distCoeffs=None, flags=cv2.SOLVEPNP_P3P)
        if not retval:
            return None
        ext_l0_l1 = utils.rodrigues_to_mat(rvec, tvec)  # extrinsic (4,4) FROM world_left_0 to world_left_1 (camera)
        return ext_l0_l1

    def pnp_ransac(self):
        eps = 0.99 # initial percent of outliers
        s=4 # number of points to estimate
        p=0.999 # probability we want to get a subset of all inliers
        iters_to_do = np.log(1-p) / np.log(1-(1-eps)**s)
        iters_done = 0

        best_ext_l0_l1 = None
        best_inliers_bool = np.zeros(1)
        best_proj_errors = None

        while iters_done <= iters_to_do:
            ext_l0_l1 = self.pnp()
            if ext_l0_l1 is None: continue
            inliers_bool, projections_errors_to_l1 = utils.get_consistent_with_extrinsic(self.kp_l1, self.kp_r1, self.pc_l0_r0, ext_l0_l1,self.ext_li_ri, self.k)
            inlier_per = sum(inliers_bool) / self.kp_l1.shape[1]
            eps = min(eps,1-inlier_per) # get upper bound on percent of outliers
            iters_done += 1
            iters_to_do = np.log(1-p) / np.log(1-(1-eps)**s) # re-compute required number of iterations
            if sum(inliers_bool) >= sum(best_inliers_bool):
                best_ext_l0_l1 = ext_l0_l1
                best_inliers_bool = inliers_bool
                best_proj_errors = projections_errors_to_l1

        # refine ext_l0_l1 by computing it from all its inliers
        inlier_pc_l0 = self.pc_l0_r0[:, best_inliers_bool]
        inlier_kp_l1 = self.kp_l1[:, best_inliers_bool]
        try:
            tmp_pc_l0, tmp_pxls_l1 = get_pc_pxls_for_cv_pnp(pc_l0_r0=inlier_pc_l0, pxls_l1=inlier_kp_l1, size=inlier_pc_l0.shape[1])
            retval, rvec, tvec = cv2.solvePnP(objectPoints=tmp_pc_l0, imagePoints=tmp_pxls_l1, cameraMatrix=self.k3, distCoeffs=None)
            best_ext_l0_l1 = utils.rodrigues_to_mat(rvec, tvec)  # extrinsic (4,4) from world_left_0 to world_left_1
            best_inliers_bool, best_proj_errors = utils.get_consistent_with_extrinsic(self.kp_l1, self.kp_r1,
                                                                                         self.pc_l0_r0, best_ext_l0_l1,
                                                                                         self.ext_li_ri, self.k)
        except:
            print("failure in refine best_ext_l1")

        self.best_proj_errors = best_proj_errors
        self.best_ext_l0_l1 = best_ext_l0_l1
        self.best_inliers_bool = best_inliers_bool
        return self.best_ext_l0_l1, self.best_inliers_bool, self.best_proj_errors

def get_pc_pxls_for_cv_pnp(pc_l0_r0, pxls_l1, size):
    """ make sure that pc_l0_r0[i] is the world_point of pxls_l1[i]
    :param pc_l0_r0: (3,n) object points
    :param pxls_l1: (2,n) image points
    :param size: number of points to use for pnp
    :return: pc_pnp_l0_r0 (size,1,3)
             pxls_pnp_l1 (size,1,2)
    """
    assert size <= pxls_l1.shape[1]
    assert pxls_l1.shape[1] == pc_l0_r0.shape[1]

    if size == pxls_l1.shape[1]:
        pc_l0 = np.ascontiguousarray(pc_l0_r0.T).reshape((size, 1, 3))
        pxls_l1 = np.ascontiguousarray(pxls_l1.T).reshape((size, 1, 2))
        return pc_l0, pxls_l1

    # else, size < num_of_4_matched points
    indices = np.random.choice(pxls_l1.shape[1],size=size, replace=False)
    pc_l0 = pc_l0_r0[:,indices]
    pxls_l1 = pxls_l1[:,indices]
    pc_l0 = np.ascontiguousarray(pc_l0.T).reshape((size, 1, 3))
    pxls_l1 = np.ascontiguousarray(pxls_l1.T).reshape((size, 1, 2))
    return pc_l0, pxls_l1

#########################   Visualization utils  #########################
def plot_inliers_outliers_of_ext1(idx_of_l0, filt_kp_l0, filt_kp_l1, ext_l1_inliers_bool):
    """ :param filt_kp_l0/1: (2,n)
    :param ext_l1_inliers_bool: boolean array of size n """
    img_l0, img_r0 = kitti.read_images(idx=idx_of_l0)
    img_l1, img_r1 = kitti.read_images(idx=idx_of_l0+1)
    # plot inliers and outliers of ext_l1 among l0 and l1
    img_l0 = cv2.cvtColor(img_l0, cv2.COLOR_GRAY2BGR)
    for i, px in enumerate(filt_kp_l0.astype(int).T):
        color = utils.INLIER_COLOR if ext_l1_inliers_bool[i] else utils.OUTLIER_COLOR
        img_l0 = cv2.circle(img_l0, center=tuple(px), radius=4, color=color, thickness=1)
    utils.cv_disp_img(img_l0, title=f"l0 {idx_of_l0} inlier (green),outlier (blue) of best ext_l1 ",save=False)

    # plot inliers and outliers of ext_l1 among l0 and l1
    img_l1 = cv2.cvtColor(img_l1, cv2.COLOR_GRAY2BGR)
    for i, px in enumerate(filt_kp_l1.astype(int).T):
        color = utils.INLIER_COLOR if ext_l1_inliers_bool[i] else utils.OUTLIER_COLOR
        img_l1 = cv2.circle(img_l1, center=tuple(px), radius=4, color=color, thickness=1)
    utils.cv_disp_img(img_l1, title=f"l1 {idx_of_l0+1} inlier (green),outlier (blue) of best ext_l1 ", save=False)

def draw_inliers_double(l0_idx, kp1, kp2, inliers, best_proj_errors, size, save=False):
    """  :param kp1/2: (2,n)
    :param inliers: bool array of size (n) """
    img_l0, img_r0 = kitti.read_images(idx=l0_idx, color_mode=cv2.IMREAD_COLOR)
    img_l1, img_r1 = kitti.read_images(idx=l0_idx + 1, color_mode=cv2.IMREAD_COLOR)
    proj_errors_l1 = best_proj_errors
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle(f"l{l0_idx}-l{l0_idx+1}, {sum(inliers)} inliers / {inliers.size} matches")
    ax1.imshow(img_l0); ax1.axis('off')
    ax2.imshow(img_l1); ax2.axis('off')
    inds = range(kp1.shape[1])
    if size:
        inds = np.random.choice(kp1.shape[1], size=size, replace=False)
    for i in inds:
        xy1 = kp1[:,i]
        xy2 = kp2[:,i]
        print(f'{xy1}-{xy2} {"inlier" if inliers[i] else "outlier"}, {proj_errors_l1[i]:.2f}')
        color = "green" if inliers[i] else "red"
        con = ConnectionPatch(xyA=tuple(xy1), xyB=tuple(xy2), coordsA="data", coordsB="data",
                              axesA=ax1, axesB=ax2, color=color)
        ax2.add_artist(con)
        c = "g" if inliers[i] else "r"
        ax1.plot(xy1[0], xy1[1], f'{c}o', markersize=3)
        ax2.plot(xy2[0], xy2[1], f'{c}o', markersize=3)
    fig.subplots_adjust(left=0.01, bottom=0.19, right=0.99, top=0.94, wspace=0.01, hspace=0.2)
    print(f'---- end l{l0_idx}-l{l0_idx+1} ------- ')
    if save:
        path = os.path.join(utils.fig_path(), 'tmp', f'{l0_idx}_{l0_idx + 1}' + '.png'); path = utils.get_avail_path(path)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()