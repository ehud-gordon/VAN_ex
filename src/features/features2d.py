""" Detect keypoints, compute descriptors and  matching. Supports filtering of matches, uses cv2. """
import cv2
import numpy as np

from itertools import compress

import utils
from .plot import DrawMatchesDouble
import utils.array

MATCH_Y_DIST_MAX = 2

class Features:
    def __init__(self, detector="SIFT", descriptor="SIFT", matcher="BF", grid=False, save=False, **kwargs):
        self.detector = detector # str
        self.descriptor = descriptor # str
        self.matcher_type = matcher # str
        self.grid = grid # bool
        self.save = save # bool
        self.plot_keypoints = False
        self.plot_matches = False
        self.plot_grid = False
        self.matcher = self.decide_matcher()

    def decide_matcher(self):
        if self.matcher_type == "BF":
            if self.descriptor in ["SURF", "SIFT"]:
                return cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
            if self.descriptor in ["ORB", "BRISK", "BRIEF", "AKAZE"]:
                return cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)

        elif self.matcher_type == "FLANN":
            index_params = None
            search_params = dict(checks=100)
            if self.descriptor in ["SURF", "SIFT"]:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            elif self.descriptor in ["ORB", "BRISK", "BRIEF", "AKAZE"]:
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                    table_number=6,  # 12
                                    key_size=12,  # 20
                                    multi_probe_level=1)  # 2
            return cv2.FlannBasedMatcher(index_params, search_params)

    def keypoints_descriptors_grid(self, img):
        """ divides an image into cells and finds keypoints and descriptors for each cell.
        returns combined keypoints and descriptors for the entire image. """
        grid = utils.image.image_to_grid(img)
        grid_keypoints = []
        grid_descriptors = []
        for cell, origin in grid:
            cell_keypoints, cell_descriptors = self.detectAndCompute(cell, plot_keypoints=self.plot_grid) # (2,n) in (x,y) format
            if cell_keypoints.size==0:
                continue
            orig_y, orig_x = origin[0], origin[1] # [y,x]
            corrected_keypoints = cell_keypoints + np.array([[orig_x], [orig_y]])
            grid_descriptors.append(cell_descriptors)
            grid_keypoints.append(corrected_keypoints)

        grid_keypoints = np.concatenate(grid_keypoints, axis=1)
        grid_descriptors = np.concatenate(grid_descriptors, axis=1)
        print(f"num of combined keypoints in all grid:{grid_keypoints.shape[1]}")
        if self.plot_keypoints:
            cv2_keypoints= [cv2.KeyPoint(x,y,7) for x,y in grid_keypoints.T]
            img = cv2.drawKeypoints(img, cv2_keypoints, outImage=None, color=(255, 0, 0), flags=0)
            utils.image.cv_show_img(img, title=f"{self.detector}_{self.descriptor}_keypoints", save=False)

        return grid_keypoints, grid_descriptors

    def detectAndCompute(self, img, plot_keypoints=False):
        """ Detect keypoints in image and compute their descriptors.

        :param bool plot_keypoints: whether to output a plot of the detected keypoints
        :return:
            keypoints: (2,n) ndarray of format (x,y) of the detected pixels
            descriptors: (m,n) ndarray, where m is descriptor size (128), and n is number of detected keypoints
        """
        if self.detector == "SURF" and self.descriptor == "SURF":
            surf_feature2d = cv2.xfeatures2d_SURF.create(hessianThreshold=400) # default 100
            keypoints, descriptors = surf_feature2d.detectAndCompute(img, None) # SURF needs L2 NORM

        elif self.detector == "AKAZE" and self.descriptor == "AKAZE":
            AKAZE = cv2.AKAZE_create(threshold=0.001) # default 0.001f
            keypoints, descriptors = AKAZE.detectAndCompute(img, None)

        elif self.detector == "SIFT" and self.descriptor == "SIFT":
            sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10) # nfeatures=0
            # sift = cv2.SIFT_create(contrastThreshold=0.03, edgeThreshold=12)  # nfeatures=0
            keypoints, descriptors = sift.detectAndCompute(img,None)
        elif self.detector == "STAR" and self.descriptor == "BRIEF":
            star = cv2.xfeatures2d.StarDetector_create(responseThreshold=10) # default 30,
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            keypoints = star.detect(img, None)
            keypoints, descriptors = brief.compute(img, keypoints)
        else:
            raise Exception

        if plot_keypoints:
            img = cv2.drawKeypoints(img, keypoints, outImage=None, color=(255, 0, 0), flags=0)
            utils.image.cv_show_img(img, title=f"{self.detector}_{self.descriptor}_keypoints", save=False)

        keypoints = np.array([keypoint.pt for keypoint in keypoints]).T # (2,n)
        return keypoints, descriptors.T # (2, n), # (128, n)

    def detectComputeMatch(self, img1, img2, is_stereo):
        """ finds keypoints and descriptors for two images, matches between them, and
        returns keypoints and desciprotrs of (and only of) good matches

        :param bool is_stereo: are the images rectified and from stereo setup
        :returns: (keypoints1, descriptors1, keypoints2) 
            keypoints1 - (2,n) ndarray of format (x,y) of the matched keypoints in image 1
            descriptors1 - (m,n) ndarray, where m is descriptor size (128), and n is number of matched keypoints
            keypoints2 - (2,n) ndarray of format (x,y) of the matched keypoints in image 2
        """
        if self.grid:
            keypoints1, descriptors1 = self.detectAndCompute(img=img1) # (2, n1), ndarray (32, n1)
            keypoints2, descriptors2 = self.detectAndCompute(img=img2) # # (2, n2), ndarray (32, n2)
        else:
            keypoints1, descriptors1 = self.detectAndCompute(img=img1, plot_keypoints=self.plot_keypoints)  # (2, n1), ndarray (32, n1)
            keypoints2, descriptors2 = self.detectAndCompute(img=img2, plot_keypoints=self.plot_keypoints)  # (2, n2), ndarray (32, n2)

        matches = self.Match(keypoints1, descriptors1, keypoints2, descriptors2, is_stereo=is_stereo)

        keypoints1_matched, descriptors1_matched, keypoints2_matched = filter_with_matches(matches, [keypoints1, descriptors1],[keypoints2])
        if self.plot_matches:
            fig_drawer = DrawMatchesDouble(img1, img2, keypoints1_matched, keypoints2_matched)
            fig_drawer.draw_matches_double(0, save=False, title=f'{self.matcher_type}_{self.detector}_{self.descriptor}_is_stereo={is_stereo}')

        return keypoints1_matched, descriptors1_matched, keypoints2_matched

    def Match(self, keypoints1, descriptors1, keypoints2, descriptors2, is_stereo):
        """ Match between two images. Filters out bad matches"""
        matches = self.matcher.match(descriptors1.T, descriptors2.T)  # list of matches [DMatch1,... DMatch1N]
        return filter_matches(matches, keypoints1, keypoints2, is_stereo)

def filter_matches(matches, keypoints1, keypoints2, is_stereo):
    """ Filter out matches with match-distance in the upper 2 percentile.
    If keypoints come from stereo images, also filters matches where the difference in the
    y pixel is larger than a threshold.

    :param matches: list of matches [DMatch1, ... , DMatch_n]
    :param keypoints1: (2,n) ndarray of format (x,y) of the detected pixels in image 1
    :param keypoints2: (2,n) ndarray of format (x,y) of the detected pixels in image 2
    :param is_stereo: boolean, True in order to apply stereo-based filtering
    :return:
    """
    good_matches = []
    match_distances = []
    for m in matches:
        y_dist = abs(keypoints1[1, m.queryIdx] - keypoints2[1, m.trainIdx])
        if is_stereo and (y_dist > MATCH_Y_DIST_MAX):
            continue
        match_distances.append(m.distance)
        good_matches.append(m)
    # filter based on match-distance
    match_distances = np.asarray(match_distances)
    bool_of_largest = utils.array.get_perc_largest_indices(match_distances, 0.02)
    matches = list(compress(good_matches, ~bool_of_largest))

    return matches

def filter_with_matches(matches, query_lst, train_lst):
    """ Filter numpy arrays using matches.

    :param matches: a list of matches [DMatch1, ..., DMatch_N]
    :param query_lst: list of numpy arrays of the queryIdx side
    :param train_lst: list of numpy arrays of the trainIdx side
    :return: list contaning all filtered arrays
    """
    query_inds = [m.queryIdx for m in matches]
    train_inds = [m.trainIdx for m in matches]
    res = [l[:,query_inds] for l in query_lst]
    res += [l[:,train_inds] for l in train_lst]
    return res
