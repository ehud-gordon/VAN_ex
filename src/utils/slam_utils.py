""" Utility methods for use with StereoSLAM"""
import cv2
import numpy as np
from scipy.sparse.csgraph import dijkstra

import factor_graph.gtsam_utils as g_utils
from utils.shortest_path import shortest_path


from utils import kitti

########### UTILS ###############

def get_stereo_images(idx, dataset_path=None, color_mode=cv2.IMREAD_GRAYSCALE):
    """ read a pair of stereo images from kitti """
    img_left, img_right = kitti.read_images(idx, dataset_path, color_mode)
    return img_left, img_right

def get_shortest_path(j, k, distance_matrix):
    predecessors = dijkstra(distance_matrix.tocsr(), directed=False, indices=j, return_predecessors=True)[1]
    return shortest_path(j, k, predecessors)

def mahalanobis(j, k, poses, marginals, path):
    """ compute Mahalanobis distance between poses j and k using (shortest) path

    :param poses: Poses object
    :param marginals: Marginals object
    :param path: (shortest) path from j to k, along which we compute the relative pose and covariance
    """
    pose_from_j_to_k = poses.get_path_pose_from(j).to(k).along_path(path)
    t2v = g_utils.t2v(pose_from_j_to_k)
    cov_j_conditional_on_k = marginals.get_path_cov_of(j).conditional_on(k).along_path(path)
    mahal_dist = t2v.T @ cov_j_conditional_on_k @ t2v
    return mahal_dist.item()

def filter_stereo_features(sf):
    """ Detect 3D points whose location is unlikely, for our kitti 05 setup

    :param sf: StereoFeatures, contaning 2D keypoints + point-cloud
    :return: StereoFeatures without unlikely points
    """
    pc = sf.pc # point-cloud
    x_inliers = (np.abs(pc[0]) <= 30)
    y_inliers = (np.abs(pc[1]) <= 30)
    z_inliers = (pc[2] > 1) * (pc[2] < 200)
    inliers = x_inliers * y_inliers * z_inliers
    return sf.filter_with_bool(inliers)
