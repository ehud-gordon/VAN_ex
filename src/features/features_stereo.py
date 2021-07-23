""" StereoFeatures include the 2D keypoints and 3D point-cloud of a pair of stereo images"""

import numpy as np
from dataclasses import dataclass
from calib3d.triangulate import triangulate
from features.features2d import filter_with_matches
import utils.array

@dataclass
class StereoFeatures:
        idx: int  # frame index
        keypoints_left: np.ndarray # (2,n) ndarray
        descriptors_left: np.ndarray # (2,n) ndarray
        keypoints_right: np.ndarray # (2,n) ndarray
        pc : np.ndarray = None # point cloud, (3,n)
        
        def filter_with_bool(self, bool_array):
            """ filter points with boolean array of size n """
            assert len(bool_array) == self.keypoints_left.shape[1]
            kp_left, desc_left, kp_right, pc = \
                utils.array.filter_(bool_array, self.keypoints_left, self.descriptors_left, self.keypoints_right, self.pc)
            return StereoFeatures(self.idx, kp_left, desc_left, kp_right, pc)

        def filter_with_matches(self, matches, is_query):
            """ filter StereoFeatures using matches from cv2.Match

            :param matches: list of matches [DMatch1, ..., DMatch_N ] between the two frames
            :param is_query: bool, whether we filter with queryIdx or trainIdx of matches.
            :return: StereoFeatures containing only matched points
            """
            if is_query:
                kp_left, desc_left, kp_right, pc = \
                    filter_with_matches(matches, [self.keypoints_left, self.descriptors_left, self.keypoints_right, self.pc], [])
            else: # is train
                kp_left, desc_left, kp_right, pc =\
                    filter_with_matches(matches,[], [self.keypoints_left, self.descriptors_left, self.keypoints_right, self.pc])
            return StereoFeatures(self.idx, kp_left, desc_left, kp_right, pc)


def compute_stereo_features(left_image, right_image, features, k, ext_l_to_r, frame_idx):
    """ Finds StereoFeatures (2D keypoints + point-cloud) in a pair of left and right stereo images.

    :param features: Features object, used for keypoint detection and matching.
    :param k: intrinsics camera matrix (3,4) ndarray
    :param ext_l_to_r: extrinsics matrix from left camera to right camera
    :param int frame_idx: frame index of stereo images
    :return: StereoFeatures object containing keypoints and cloud-point of frame-idx
    """
    # find matching keypoints between left and right images
    keypoints_left, descriptors_left, keypoints_right = features.detectComputeMatch(left_image, right_image, is_stereo=True)
    # compute point-cloud
    ext_id = np.diag([1,1,1,1]) # the (4,4) identity matrix
    pc = triangulate(keypoints_left, keypoints_right, k, ext_id, ext_l_to_r)
    return StereoFeatures(frame_idx, keypoints_left, descriptors_left, keypoints_right, pc)

def match_two_stereo_pairs(features, sf1, sf2):
    """ Matches between two StereoFeatures objects, based on left image descriptors.

    :param features: Features object, used for keypoint detection and matching.
    :param sf1: StereoFeatures of stereo pair at index 1
    :param sf2: StereoFeatures of stereo pair at index 2
    :return:
        matches - list of matches [DMatch1, ..., DMatch_N] between the two frames
        sf1_matched: the StereoFeatures of pair 1 that are matched to StereoFeatures of pair 2
        sf2_matched: the StereoFeatures of pair 2 that are matched to StereoFeatures of pair 2
    """
    matches = features.Match(sf1.keypoints_left, sf1.descriptors_left, sf2.keypoints_left, sf2.descriptors_left, is_stereo=False)
    sf1_matched = sf1.filter_with_matches(matches, is_query=True)
    sf2_matched = sf2.filter_with_matches(matches, is_query=False)
    return matches, sf1_matched, sf2_matched

