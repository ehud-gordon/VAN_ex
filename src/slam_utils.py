import cv2
import numpy as np
from utils.array import filter_
from features.stereo_features import StereoFeatures
from features.features2d import filter_with_matches
from calib3d.triangulate import triangulate
import pgm.utils as g_utils
from utils.shortest_path import shortest_path

import kitti

def get_stereo_images(idx, dataset_path=None, color_mode=cv2.IMREAD_GRAYSCALE):
    img_left, img_right = kitti.read_images(idx, dataset_path, color_mode)
    return img_left, img_right

def compute_stereo_features(frame_idx, features, dataset_path, k, ext_l_to_r):
    left_img, right_img = get_stereo_images(frame_idx, dataset_path)
    keypoints_left, descriptors_left, keypoints_right = features.detectComputeMatch(left_img, right_img, is_stereo=True)
    ext_id = np.diag([1,1,1,1]) # the (4,4) identity matrix
    pc = triangulate(keypoints_left, keypoints_right, k, ext_id, ext_l_to_r) # point-cloud
    pc_filter = relative_inliers(pc)
    keypoints_left, descriptors_left, keypoints_right, pc = \
        filter_(pc_filter, keypoints_left, descriptors_left, keypoints_right, pc)

    stereo_features = StereoFeatures(frame_idx, keypoints_left, descriptors_left, keypoints_right, pc)
    return stereo_features

def match_two_stereo_pairs(features, sf1, sf2):
    matches = features.Match(sf1.keypoints_left, sf1.descriptors_left, sf2.keypoints_left, sf2.descriptors_left, is_stereo=False)
    kp_left_1_matched, kp_right_1_matched, pc_2_matched = filter_with_matches(matches,
                                                                              [sf1.keypoints_left, sf1.keypoints_right],
                                                                              [sf2.pc])
    return matches, kp_left_1_matched, kp_right_1_matched, pc_2_matched


def compute_mahalanobis_distance(j, k, poses, marginals, predecessors):
    """ compute Mahalnobis distance between poses j and k using shortest path """
    path_from_j_to_k = shortest_path(j, k, predecessors)
    # TODO add handling with reverse paths 
    pose_from_j_to_k = poses.get_path_pose_from(j).to(k).along_path(path_from_j_to_k)
    t2v = g_utils.t2v(pose_from_j_to_k)
    cov_j_conditional_on_k = marginals.get_path_cov_of(j).conditional_on(k).along_path(path_from_j_to_k)
    mahal_dist = t2v.T @ cov_j_conditional_on_k @ t2v
    return mahal_dist.item()

def relative_inliers(pc):
    """ :param pc: (3,n) """
    x_abs = np.abs(pc[0]); y_abs = np.abs(pc[1])
    x_crit = (x_abs <= 30)
    y_crit = (y_abs <= 30)
    z_crit1 = pc[2] < 200
    z_crit2 = pc[2] > 1
    z_crit = z_crit1 * z_crit2
    inliers = (x_crit * y_crit) * z_crit
    return inliers

def quantile_inliers(pc, q=0.99):
    pc_abs = np.abs(pc)
    x_quant, y_quant, z_quant = np.quantile(pc_abs, q=q, axis=1)
    x_crit = (pc_abs[0] <= x_quant)
    y_crit = (pc_abs[1] <= y_quant)
    z_crit = (pc_abs[2] <= z_quant)
    inliers = x_crit * y_crit * z_crit
    return inliers