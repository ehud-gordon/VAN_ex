"""
Performs Perspective-n-Points, finds extrinsics matrix from world objects to matched pixels.
"""
import numpy as np
import cv2

import utils

def pnp(k, pc, pixels, size):
    """ Computes extrinsics matrix from world to camera.

    :param pc: point cloud in world CS
    :param pixels: matching pixels of points in pc
    :param size: number of points to use for cv2.SolvePnP
    :return: the extrinsics matrix from the world CS to the camera CS
    """
    assert size>=4
    k3 = k[:,:3] # (3,3)
    pc_pnp, pixels_pnp = get_pc_pixels_for_cv_pnp(pc, pixels, size=size)
    # object (world) points are in world CS, returns rotation and translation from world (object) to cam.
    try:
        if size == 4:
            retval, rvec, tvec = cv2.solvePnP(objectPoints=pc_pnp, imagePoints=pixels_pnp, cameraMatrix=k3, distCoeffs=None, flags=cv2.SOLVEPNP_P3P)
        else:
            retval, rvec, tvec = cv2.solvePnP(objectPoints=pc_pnp, imagePoints=pixels_pnp, cameraMatrix=k3, distCoeffs=None)
    except:
        return None
    if retval == False:
        return None
    ext_world_to_cam = utils.geometry.rodrigues_to_mat(rvec, tvec)  # extrinsic (4,4) from WORLD to CAMERA
    return ext_world_to_cam

def pnp_stereo_ransac(pixels_left, pixels_right, pc, k, ext_l_to_r, max_iters=np.inf):
    """ performs PnP with consensus matching, by defininig inliers as pixels whose 3D locations are projected correctly
    both to pixels_left and pixels_right.

    :param pixels_left: (2,n) pixels in image1 matched to pixels2
    :param pixels_right: (2,n) pixels in image2 matched to pixels1
    :param pc: (3,n) point cloud of pixels
    :param k: intrinsics camera matrix (shared by both cameras)
    :param ext_l_to_r: extrinsics matrix from left to right
    :param int max_iters: maximum number of ransac iterations to perform
    :return:
            ext_world_to_cam: (4,4)
            inliers_bool: (n,) array with True in the indices of inliers
            inliers_percent: percent of PnP inliers, between 0 and 1
    """
    assert pixels_left.shape[1] == pixels_right.shape[1] == pc.shape[1]

    eps = 0.99 # initial percent of outliers
    s=4 # number of points to estimate
    p=0.9999 # probability we want to get a subset of all inliers
    iters_to_do = np.log(1-p) / np.log(1-(1-eps)**s)
    iters_done = 0

    best_ext_world_to_cam = None
    best_inliers_bool = np.zeros(1)
    best_inliers_percent = 0
    while iters_done <= min(iters_to_do, max_iters):
        ext_world_to_cam = pnp(k, pc, pixels_left, size=4)
        if ext_world_to_cam is None: continue
        inliers_bool = utils.geometry.get_extrinsics_inliers_stereo(pixels_left, pixels_right, pc, ext_world_to_cam, ext_l_to_r, k)
        inliers_percent = sum(inliers_bool) / pixels_left.shape[1]
        eps = min(eps,1-inliers_percent) # get upper bound on percent of outliers
        iters_done += 1
        iters_to_do = np.log(1-p) / np.log(1-(1-eps)**s) # re-compute required number of iterations
        # if a better results was achieved, update our current best
        if inliers_percent >= best_inliers_percent:
            best_ext_world_to_cam = ext_world_to_cam
            best_inliers_bool = inliers_bool
            best_inliers_percent = inliers_percent

    # attempt to refine ext_world_to_cam by using all its inliers
    num_inliers_before_refine = sum(best_inliers_bool)
    if num_inliers_before_refine < 50: # skip refining
        return best_ext_world_to_cam, best_inliers_bool, best_inliers_percent

    pc_inliers = pc[:, best_inliers_bool]
    pixels_left_inliers = pixels_left[:, best_inliers_bool]

    ext_world_to_cam_refined = pnp(k, pc_inliers, pixels_left_inliers, size=pixels_left_inliers.shape[1])
    if ext_world_to_cam_refined is None: # if refining failed, return previous results
        return best_ext_world_to_cam, best_inliers_bool, best_inliers_percent

    # Verifing that refining improved the extrinsics matrix
    rot_diff, trans_diff = utils.geometry.compare_ext_mats(ext_world_to_cam_refined, best_ext_world_to_cam)
    inliers_bool_refined = utils.geometry.get_extrinsics_inliers_stereo(pixels_left, pixels_right, pc,
                                                                        ext_world_to_cam_refined, ext_l_to_r, k)
    num_inliers_after_refine = sum(inliers_bool_refined)
    inliers_percent_after_refine = num_inliers_after_refine / pixels_left.shape[1]
    # this is a strange bug the resulting ext is wildly incorrect
    if num_inliers_after_refine < 50 or rot_diff > 1 or trans_diff > 2:
        return best_ext_world_to_cam, best_inliers_bool, best_inliers_percent
    return ext_world_to_cam_refined, inliers_bool_refined, inliers_percent_after_refine

def get_pc_pixels_for_cv_pnp(pc, pixels, size):
    """ reshape np arrays for use in PnP.
    make sure that pc[i] is the world_point of pxls[i]

    :param pc: (3,n) ndarray object points
    :param pixels: (2,n) ndarray image points
    :param int size: number of points to use for pnp
    :return: pc_size (size,1,3)
             pixels_size (size,1,2)
    """
    assert 4 <= size <= pixels.shape[1]
    assert pixels.shape[1] == pc.shape[1]

    # reshape pc and pixels for use in pnp
    if size == pixels.shape[1]:
        pc_size = np.ascontiguousarray(pc.T).reshape((size, 1, 3))
        pixels_size = np.ascontiguousarray(pixels.T).reshape((size, 1, 2))
        return pc_size, pixels_size

    # else, choose (size) random points
    indices = np.random.choice(pixels.shape[1], size=size, replace=False)
    pc_size = pc[:,indices]
    pixels_size = pixels[:, indices]
    pc_size = np.ascontiguousarray(pc_size.T).reshape((size, 1, 3))
    pixels_size = np.ascontiguousarray(pixels_size.T).reshape((size, 1, 2))
    return pc_size, pixels_size
