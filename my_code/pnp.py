import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import os

import kitti
import utils

def pnp(k, pc, kp, size):
    """
    :param pc: in world CS
    :param kp: matching pixels of points in pc
    :param size: number of points to use for cv2.SolvePnP
    :return: ext_world_to_cam
    """
    assert size>=4
    k3 = k[:,:3] # (3,3)
    pc_pnp, pxls_pnp = get_pc_pxls_for_cv_pnp(pc, kp, size=size)
    # object (world) points are in world CS, returns rotation and translation from world (object) to cam.
    exception_raised = False
    retval = True
    if size == 4:
        retval, rvec, tvec = cv2.solvePnP(objectPoints=pc_pnp, imagePoints=pxls_pnp, cameraMatrix=k3, distCoeffs=None, flags=cv2.SOLVEPNP_P3P)
    else:
        try:
            retval, rvec, tvec = cv2.solvePnP(objectPoints=pc_pnp, imagePoints=pxls_pnp, cameraMatrix=k3, distCoeffs=None)
        except:
            exception_raised = True
    if (retval is False) or exception_raised:
        return None
    ext_world_to_cam = utils.rodrigues_to_mat(rvec, tvec)  # extrinsic (4,4) from WORLD (left_0) to CAMERA (left_j)
    return ext_world_to_cam

def pnp_ransac(kp_l, kp_r, pc, k, ext_l_to_r, frame="", max_iters=np.inf):
    """
    :param pc: in world CS. returns world_to_cam
    """
    eps = 0.99 # initial percent of outliers
    s=4 # number of points to estimate
    p=0.9999 # probability we want to get a subset of all inliers
    iters_to_do = np.log(1-p) / np.log(1-(1-eps)**s)
    iters_done = 0

    best_ext_w_to_c = None
    best_inliers_bool = np.zeros(1)
    best_proj_errors_l, best_proj_errors_r = None, None
    while iters_done <= min(iters_to_do, max_iters):
        ext_w_to_c = pnp(k, pc, kp_l, size=4)
        if ext_w_to_c is None: continue
        inliers_bool, proj_errors_l, proj_errors_r = utils.get_consistent_with_extrinsic(kp_l, kp_r, pc,
                                                                                ext_w_to_c, ext_l_to_r, k)
        inlier_perc = sum(inliers_bool) / kp_l.shape[1]
        eps = min(eps,1-inlier_perc) # get upper bound on percent of outliers
        iters_done += 1
        iters_to_do = np.log(1-p) / np.log(1-(1-eps)**s) # re-compute required number of iterations
        if sum(inliers_bool) >= sum(best_inliers_bool):
            best_ext_w_to_c = ext_w_to_c
            best_inliers_bool = inliers_bool
            best_proj_errors_l = proj_errors_l
            best_proj_errors_r = proj_errors_r

    # attempt to refine ext_w_to_c by computing it from all its inliers
    num_points = kp_l.shape[1]
    num_inliers_before_ref = sum(best_inliers_bool); perc_inliers_before_ref = f'{num_inliers_before_ref / num_points:.1%}'
    # print(f'pnp_ransac() did {iters_done} iters. before: {num_inliers_before_ref}/{self.kp_l.shape[1]}={perc_inliers_before_ref}')
    pc_inliers = pc[:, best_inliers_bool]
    kp_l_inliers = kp_l[:, best_inliers_bool]
    if num_inliers_before_ref < 50: # skip refining
        return best_ext_w_to_c, best_inliers_bool, best_proj_errors_l, best_proj_errors_r
    ext_w_to_c_ref = pnp(k, pc_inliers, kp_l_inliers, size=kp_l_inliers.shape[1]) 
    if ext_w_to_c_ref is None:
        print(f'frame={frame}, error in pnp_ransac() when refining, before: {num_inliers_before_ref}/{num_points}={perc_inliers_before_ref}')
        return best_ext_w_to_c, best_inliers_bool, best_proj_errors_l, best_proj_errors_r
    
    rot_diff, trans_diff = utils.compare_ext_mat(ext_w_to_c_ref, best_ext_w_to_c) # rot in degrees
    inliers_bool_ref, proj_errors_l_ref, proj_errors_r_ref = utils.get_consistent_with_extrinsic(kp_l, kp_r,
                                                                                    pc, ext_w_to_c_ref,
                                                                                    ext_l_to_r, k)
    num_inliers_after_ref = sum(inliers_bool_ref); perc_after_ref = f'{num_inliers_after_ref/num_points:.1%}'

    # this is a strange bug where retval==True, but the resulting ext is wildly incorrect
    if num_inliers_after_ref < 50 or rot_diff > 1 or trans_diff > 2:
        print(f'frame={frame}, pnp refine bug. before: {num_inliers_before_ref}/{num_points}={perc_inliers_before_ref} ' +
              f'after: {num_inliers_after_ref}/{num_points}={perc_after_ref} ' +
              f'rot_diff:{rot_diff:.1f} deg, trans_diff={trans_diff:.1f} meters')
        return best_ext_w_to_c, best_inliers_bool, best_proj_errors_l, best_proj_errors_r
    
    return ext_w_to_c_ref, inliers_bool_ref, proj_errors_l_ref, proj_errors_r_ref

def pnp_inliers():
    pass

def get_pc_pxls_for_cv_pnp(pc, pxls, size):
    """ reshape np arrays for use in pnp.
    make sure that pc[i] is the world_point of pxls[i]
    :param pc: (3,n) object points
    :param pxls: (2,n) image points
    :param size: number of points to use for pnp
    :return: pc_size (size,1,3)
             pxls_size (size,1,2)
    """
    assert 4 <= size <= pxls.shape[1]
    assert pxls.shape[1] == pc.shape[1]

    # reshape pc and pxls for use in pnp
    if size == pxls.shape[1]:
        pc_size = np.ascontiguousarray(pc.T).reshape((size, 1, 3))
        pxls_size = np.ascontiguousarray(pxls.T).reshape((size, 1, 2))
        return pc_size, pxls_size

    # else, choose (size) random points
    indices = np.random.choice(pxls.shape[1],size=size, replace=False)
    pc_size = pc[:,indices]
    pxls_size = pxls[:,indices]
    pc_size = np.ascontiguousarray(pc_size.T).reshape((size, 1, 3))
    pxls_size = np.ascontiguousarray(pxls_size.T).reshape((size, 1, 2))
    return pc_size, pxls_size
