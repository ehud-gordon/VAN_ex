""" Performs Perspective-n-Points, finds extrinsics matrix from world objects to matched pixels. """
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
    pc_pnp, pixels_pnp = reshape_for_cv_pnp(pc, pixels, size=size)
    # object (world) points are in world CS, returns rotation and translation from world (object) to cam.
    try:
        if size == 4:
            retval, rvec, tvec = cv2.solvePnP(objectPoints=pc_pnp, imagePoints=pixels_pnp, cameraMatrix=k3, distCoeffs=None, flags=cv2.SOLVEPNP_P3P)
        else:
            retval, rvec, tvec = cv2.solvePnP(objectPoints=pc_pnp, imagePoints=pixels_pnp, cameraMatrix=k3, distCoeffs=None)
    except:
        return None
    if not retval:
        return None
    ext_world_to_cam = utils.geometry.rodrigues_to_mat(rvec, tvec)  # extrinsic (4,4) from WORLD to CAMERA
    return ext_world_to_cam

def pnp_stereo_ransac(sf1, sf2, k, ext_l_to_r, max_iters=np.inf):
    """ performs PnP with consensus matching.
    w.r.t. some pose_from_2_to_1, inliers are defined as those satisfying the below:
    (1) their 3D locations (from sf2.pc) are projected neat to both sf1.pixels_left and sf1.pixels_right
    (2) their corrected 3D locations (sf1.pc, and sf2.pc transformed to sf1 CS) are close

    :param sf1: StereoFeatures (2D keypoints + point-cloud) of frame 1
    :param sf2: StereoFeatures (2D keypoints + point-cloud) of frame 2
    :param k: intrinsics camera matrix (shared by both cameras)
    :param ext_l_to_r: extrinsics matrix from left to right
    :param int max_iters: maximum number of ransac iterations to perform
    :return:
            pose_2_to_1: (4,4) extrinsics matrix from left_camera 2 to left_camera 1
            inliers_bool: (n,) array with True in the indices of inliers
            inliers_percent: percent of PnP inliers, between 0 and 1
    """
    pixels_left = sf1.keypoints_left # (2,n) pixels in left image 1
    pc = sf2.pc # (3,n) point-cloud, world location of pixels
    assert pixels_left.shape[1] == pc.shape[1]


    eps = 0.99 # initial percent of outliers
    s=4 # number of points to estimate
    p=0.9999 # probability we want to get a subset of all inliers
    iters_to_do = np.log(1-p) / np.log(1-(1-eps)**s)
    iters_done = 0

    best_pose_2_to_1 = None
    best_inliers_bool = np.zeros(1)
    best_inliers_percent = 0
    while iters_done <= min(iters_to_do, max_iters):
        pose_2_to_1 = pnp(k, pc, pixels_left, size=4)
        if pose_2_to_1 is None: continue
        inliers_bool = get_extrinsics_inliers_stereo(sf1, sf2, pose_2_to_1, ext_l_to_r, k)
        inliers_percent = sum(inliers_bool) / pixels_left.shape[1]
        eps = min(eps,1-inliers_percent) # get upper bound on percent of outliers
        iters_done += 1
        iters_to_do = np.log(1-p) / np.log(1-(1-eps)**s) # re-compute required number of iterations
        # if a better results was achieved, update our current best
        if inliers_percent >= best_inliers_percent:
            best_pose_2_to_1 = pose_2_to_1
            best_inliers_bool = inliers_bool
            best_inliers_percent = inliers_percent

    # attempt to refine pose_2_to_1 by using all its inliers
    num_inliers_before_refine = sum(best_inliers_bool)
    if num_inliers_before_refine < 50: # skip refining
        return best_pose_2_to_1, best_inliers_bool, best_inliers_percent

    pc_inliers = pc[:, best_inliers_bool]
    pixels_left_inliers = pixels_left[:, best_inliers_bool]

    pose_2_to_1_refined = pnp(k, pc_inliers, pixels_left_inliers, size=pixels_left_inliers.shape[1])
    if pose_2_to_1_refined is None: # if refining failed, return previous results
        return best_pose_2_to_1, best_inliers_bool, best_inliers_percent

    # Verifing that refining improved the extrinsics matrix
    rot_diff, trans_diff = utils.geometry.compare_ext_mats(pose_2_to_1_refined, best_pose_2_to_1)
    inliers_bool_refined = get_extrinsics_inliers_stereo(sf1, sf2, pose_2_to_1_refined, ext_l_to_r, k)
    num_inliers_after_refine = sum(inliers_bool_refined)
    inliers_percent_after_refine = num_inliers_after_refine / pixels_left.shape[1]
    # this is a strange bug the resulting ext is wildly incorrect
    if num_inliers_after_refine < 50 or rot_diff > 1 or trans_diff > 2:
        return best_pose_2_to_1, best_inliers_bool, best_inliers_percent
    return pose_2_to_1_refined, inliers_bool_refined, inliers_percent_after_refine

def reshape_for_cv_pnp(pc, pixels, size):
    """ reshape np arrays for use in PnP.
    make sure that pc[i] is the world point of pixels[i]

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

def get_extrinsics_inliers_stereo(sf1, sf2, pose_2_to_1, ext_l_to_r, k, threshold=2):
    """ Computes the inliers of matches between pixels and point-cloud, w.r.t. an extrinsics matrix.
    The inputs are:
    (1) matching pixels in left and right images.
    (2) their 3D locations (in frame1 and frame2 CS)
    (3) Extrinsics matrices: pose_from_2_to_1 and ext_left_to_right.

    W.r.t. some pose_from_2_to_1, inliers are defined as those satisfying the below:
    (1) their 3D locations (from sf2.pc) are projected near to both sf1.pixels_left and sf1.pixels_right
    (2) their corrected 3D locations (sf1.pc, and sf2.pc transformed to sf1 CS) are close

    :param sf1: StereoFeatures of frame 1
    :param sf2: StereoFeatures of frame2
    :param pose_2_to_1: (4,4) extrinsics matrix from left-camera-2 CS to left-camera-1 CS
    :param ext_l_to_r: (4,4) extrinsic matrix from left-camera CS to right-camera CS
    :param k: (4,4) intrinsics camera matrix
    :param threshold: L2 distance used as reprojection threshold to determine inliers.
    :return:
        inliers - boolean ndarray (n,), with True in indices of inliers
    """
    pixels_left = sf1.keypoints_left # (2,n) pixels in left image, matched to pixels_right and pc
    pixels_right = sf1.keypoints_right # (2,n) pixels in right image, matched to pixels_left and pc
    pc = sf2.pc # (3,n) point-cloud, matched to pixels_left and pixels_right
    assert pixels_left.shape[1] == pixels_right.shape[1] == pc.shape[1]
    # fix shape if needed
    if pc.shape[0] == 3:
        pc = np.vstack((pc, np.ones(pc.shape[1])))  # (4,n)

    # compute projection matrices
    projection_world_to_left = k @ pose_2_to_1  # (3,4) # from world to pixels_left
    ext_world_to_right = ext_l_to_r @ pose_2_to_1  # (4,4) # from world to camera_right
    projection_world_to_right = k @ ext_world_to_right  # (3,4) # from world to pixels_right

    # project point-cloud to left image
    projected_left = projection_world_to_left @ pc  # (3,n) inhomogeneous pixels
    projected_left = projected_left[0:2] / projected_left[-1]  # (2,n)

    # project point-cloud to right image
    projected_right = projection_world_to_right @ pc  # (3,n) inhomogeneous pixels
    projected_right = projected_right[0:2] / projected_right[-1]  # (2,n)

    projection_errors_left = np.linalg.norm((pixels_left - projected_left), axis=0) # L2 norm
    projection_errors_right = np.linalg.norm((pixels_right - projected_right), axis=0)  # L2 norm
    pixels_inliers_left = projection_errors_left <= threshold
    pixels_inliers_right = projection_errors_right <= threshold
    pixel_inliers = pixels_inliers_left * pixels_inliers_right # (n,)

    # compute point cloud inliers
    pc_2_in_CS_1 = utils.geometry.transform_pc_to_world(pose_2_to_1, pc)
    pc_l2_norm = np.linalg.norm(pc_2_in_CS_1-sf1.pc, axis=0)
    pc_inliers = pc_l2_norm < 3

    inliers = pixel_inliers * pc_inliers
    return inliers