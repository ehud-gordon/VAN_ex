"""
Performs SLAM on a set of stereo images, finds cameras locations using a combination of PnP, bundle adjustment,
pose graph and loop closures.
"""
import numpy as np
from gtsam.symbol_shorthand import X
from scipy.sparse import lil_matrix

import os
from itertools import compress
import pickle

from calib3d.triangulate import triangulate
from features.tracks import StereoTracksDB
from factor_graph.pixel_graph import PixelGraph
from factor_graph.pose_graph import PoseGraph
from core.poses import Poses
from core.covariances import Covariances
from features.features2d import Features
from features.features_stereo import StereoFeatures, match_stereo_pairs
from calib3d.pnp import pnp_stereo_ransac
import factor_graph.gtsam_utils as g_utils
from utils.slam_utils import get_stereo_images, mahalanobis, filter_stereo_features
import utils
from core.pose_vector import PoseVector

# CONSTANTS
MIN_DISTANCE_BETWEEN_KEYFRAMES = 5
MAHAL_THRESHOLD = 0.4 # Mahalanobis distance threshold
INLIERS_THRESHOLD = 0.6 # Inliers percent threshold
FRAMES_BETWEEN_LOOP_CLOSURES = 40


class StereoSLAM:
    def __init__(self, args, k, ext_l_to_r):
        """
        :param k: (3,4) ndarray, the intrinsics matrix, shared by all stereo cameras
        :param ext_l_to_r: (4,4) the extrinsics matrix from left camera to right camera
        """
        self.args = args
        self.k = k # the intrinsics matrix
        self.ext_l_to_r = ext_l_to_r # extrinsic left to right
        self.poses = Poses() # used to store and compute poses between cameras
        self.covariances = Covariances() # used to store and compute conditional covariances between two poses
        self.dataset_path = args.dataset_path
        self.keyframes = [args.startframe]
        self.frames = args.frames # list of frames indices
        self.distance_matrix = lil_matrix((len(args.frames),len(args.frames)), dtype=float)
        self.sf = dict() # StereoFeatures (keypoints + point-cloud)
        self.pose_graph = PoseGraph()
        self.tracks = StereoTracksDB() # database for storing tracks
        self.features = Features(**vars(args))  # used for keypoints detection, description and matching
        self.last_loop_closure_idx = 0


    def main(self):
        # iterate over frames, in front-end extract successive poses using PnP, in back-end build a pose graph and add loop closures
        for j in self.frames[1:]:
            i = self.frames[j-1] # i and j denote two consecutive frames/cameras, with i before j
            if i % 50 == 0: print(f"main(), {i}")
            # compute sf (stereo features: 2D keypoints + point-cloud) of frames i and j
            sf_i = self.get_stereo_features(i)
            sf_j = self.get_stereo_features(j)
            
            # match frames i and j
            matches, sf_i_matched, sf_j_matched = match_stereo_pairs(self.features,sf_i, sf_j)
            # Compute pose using PnP with RANSAC
            pose_from_j_to_i, inliers, _ = pnp_stereo_ransac(sf_i_matched, sf_j_matched, self.k, self.ext_l_to_r)
            inliers = list(compress(matches, inliers))
            # update databases with PnP result
            self.poses.set_pose_from(j).to(i).with_(pose_from_j_to_i)
            self.tracks.add(sf_i, sf_j, inliers)

            # Create new keyframe
            if self.is_keyframe(j):
                self.keyframes.append(j)
                # add an edge to Pose Graph from previous keyframe to the new keyframe
                bundle_frames = self.frames[self.keyframes[-2]:(self.keyframes[-1]+1)]
                self.add_edge_to_pose_graph(self.keyframes[-2], self.keyframes[-1], self.tracks, bundle_frames)
                self.check_loop_closures(j)
            
        # optimize pose graph and output results
        values, covariances = self.pose_graph.optimize()
        self.save_results(values, covariances)

    
    def get_stereo_features(self, idx):
        if idx in self.sf: 
            return self.sf[idx]
        left_img, right_img = get_stereo_images(idx, self.dataset_path)
        # find matching keypoints between left and right images
        keypoints_left, descriptors_left, keypoints_right =\
            self.features.detectComputeMatch(left_img, right_img, is_stereo=True)
        # compute point cloud in left camera CS
        pc = triangulate(keypoints_left, keypoints_right, self.k, np.diag([1,1,1,1]), self.ext_l_to_r)
        sf_idx = StereoFeatures(idx, keypoints_left, descriptors_left, keypoints_right, pc)
        self.sf[idx] = filter_stereo_features(sf_idx)
        return self.sf[idx]
    
    def pnp_stereo(self, sf_i, sf_j, iters=np.inf):
        """ compute pose from camera j to camera i, by matching their stereo-features (keypoints + point-cloud), and calling PnP.

        :param sf_i: StereoFeatures (2D keypoints + point-cloud) of frame i
        :param sf_j: StereoFeatures (2D keypoints + point-cloud) of frame j
        :param iters: maximum numbers of iterations while performing RANSAC
        :return:
            pose_from_j_to_i - (4,4)  extrinsics matrix from camera j to camera i
            pnp_inliers_matches_i_j - inliers from matched features between i and j w.r.t. pose_from_j_to_i
            inliers_percent - percent of PnP inliers out of matched pixels between i and j
        """
        # Match keypoints between frame i and frame j
        matches_i_j, sf_i_matched, sf_j_matched = match_stereo_pairs(self.features,sf_i, sf_j)
        # Compute PnP using RANSAC
        pose_from_j_to_i, pnp_inliers_bool, inliers_percent = pnp_stereo_ransac(sf_i_matched, sf_j_matched, self.k, self.ext_l_to_r, iters)
        pnp_inliers_matches_i_j = list(compress(matches_i_j, pnp_inliers_bool))
        return pose_from_j_to_i, pnp_inliers_matches_i_j, inliers_percent

    def is_keyframe(self, j):
        distance_to_last_keyframe = self.poses.get_distance_from(j).to(self.keyframes[-1])
        return distance_to_last_keyframe > MIN_DISTANCE_BETWEEN_KEYFRAMES
    
    def add_edge_to_pose_graph(self, frame1, frame2, tracks, frames):
        # Find pose and covariance between keyframes, using tracks to create a factor graph
        pixel_graph = PixelGraph(self.k, self.ext_l_to_r).build(frames, self.poses, tracks)
        values, marginals = pixel_graph.optimize()
        
        # extract relative pose and covariance from factor graph
        pose_from_2_to_1 = g_utils.get_pose(values, frame1, frame2)
        cov_2_conditional_on_1 = g_utils.get_conditional_covariance(marginals, frame1, frame2)
        
        # update poses, covariances and distance matrix with bundle results
        self.poses.set_pose_from(frame2).to(frame1).with_(pose_from_2_to_1)
        self.covariances.set_cov_of(frame2).conditional_on(frame1).with_(cov_2_conditional_on_1)
        self.distance_matrix[frame2, frame1] = np.linalg.det(cov_2_conditional_on_1)
        
        # add factor to pose graph
        self.pose_graph.add_factor(frame1, frame2, pose_from_2_to_1, cov_2_conditional_on_1)

    def check_loop_closures(self, j):
        """ For frame j, go over all frames k before j, attempt loop closure"""
        if self._check_preconditions_for_loop_closure(j) is False:
            return
        sf_j = self.get_stereo_features(j)
        for k in self.keyframes[0:-30]:
            # Filter candidates using Mahalanobis distance and PnP inliers percent
            # first, find shortest paths from new keyframe to all previous keyframes
            path = utils.slam_utils.get_shortest_path(j, k, self.distance_matrix)
            mahalo_distance = mahalanobis(j, k, self.poses, self.covariances, path)
            if mahalo_distance < MAHAL_THRESHOLD:
                # compute relative pose using PnP, use inliers percent as threshold
                sf_k = self.get_stereo_features(k)
                pose_from_j_to_k, inliers, inliers_percent = self.pnp_stereo(sf_k, sf_j, iters=15)
                if inliers_percent > INLIERS_THRESHOLD:
                    print(f'added LC between k={k},j={j}')
                    self.poses.set_pose_from(j).to(k).with_(pose_from_j_to_k)
                    self.last_loop_closure_idx = j
                    # add loop closure edge to pose graph
                    tracks = StereoTracksDB().add(sf_k, sf_j, inliers)
                    self.add_edge_to_pose_graph(k, j, tracks, frames=[k, j])

    
    def _check_preconditions_for_loop_closure(self, j):
        # only check for loop closures after enough frames have passed, and we haven't added one too recently
        dist_from_last_loop_closure = j - self.last_loop_closure_idx
        return len(self.keyframes) > 30 and dist_from_last_loop_closure > FRAMES_BETWEEN_LOOP_CLOSURES


    def save_results(self, values, covariances):
        # extract poses and contidional covariances
        optimal_pose_vector = PoseVector([values.atPose3(X(i)).matrix() for i in self.keyframes])
        conditional_covariances = []
        for i, j in zip(self.keyframes[:-1], self.keyframes[1:]):
            cov_j_conditional_on_i = g_utils.get_conditional_covariance(covariances, i, j)
            conditional_covariances.append(cov_j_conditional_on_i)
        # serialize
        d = {'poses':self.poses, 
             'opt_pose_vector': optimal_pose_vector,
             'conditional_covariances': conditional_covariances,
             'keyframes':self.keyframes,
             'tracks': self.tracks}

        pickle_path = os.path.join(self.args.out_dir, "poses_covariances_tracks.pkl")
        with open(pickle_path, 'wb') as handle:
            pickle.dump(d, handle)        

        
