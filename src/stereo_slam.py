"""
Performs SLAM on a set of stereo images, finds cameras locations using a combination of PnP, Bundle Adjustment,
Pose Graph and Loop Closures.
"""
import numpy as np
from gtsam.symbol_shorthand import X
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import lil_matrix
import os
from itertools import compress
import pickle
from features.tracks import StereoTracksDB
from factor_graph.pixel_graph import PixelGraph
from factor_graph.pose_graph import PoseGraph
from core.poses import Poses
from core.marginals import Marginals
from features.features2d import Features
from features.features_stereo import compute_stereo_features, match_two_stereo_pairs
from calib3d import pnp
import factor_graph.gtsam_utils as g_utils
from slam_utils import get_stereo_images, compute_mahalanobis_distance, filter_stereo_features


# CONSTANTS
MIN_DIST_BETWEEN_KEYFRAMES = 5
MAHAL_DISTANCE_THRESHOLD = 0.4
INLIERS_PERCENT_THRESHOLD = 0.6
FRAMES_BETWEEN_LOOP_CLOSURES = 40


class StereoSLAM:
    def __init__(self, args, k, ext_l_to_r):
        """
        :param k: the intrinsics matrix, shared by all stereo cameras
        :param ext_l_to_r: the extrinsics matrix from left camera to right camera
        """
        self.args = args
        self.k = k # the intrinsics matrix
        self.ext_l_to_r = ext_l_to_r # extrinsic left to right
        self.poses = Poses() # used to store and compute poses between cameras
        self.marginals = Marginals() # used to store and compute conditional covariances between two poses
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
        # iterate over frames, in frontend extract successive poses using PnP, in backend build a pose graph and add loop closures
        for j in self.frames[1:]:
            i = self.frames[j-1] # i and j denote two consecutive frames/cameras, with i before j
            if i % 30 == 0: print(f"main(), i={i}")
            # compute sf (stereo features: 2D keypoints + point-cloud) of frames i and j
            sf_i = self.get_stereo_features(i)
            sf_j = self.get_stereo_features(j)
            
            # compute pose between frames i and j using PnP, get list of PnP inliers
            pose_from_j_to_i, pnp_inliers_matches_i_j, _ = self._compute_pose_with_pnp(sf_i, sf_j)
            # update databases with PnP result
            self.poses.set_pose_from(j).to(i).with_(pose_from_j_to_i)
            self.tracks.add_frame(sf_i, sf_j, pnp_inliers_matches_i_j)
            
            # Create new keyframe
            if self.poses.get_distance_from(j).to(self.keyframes[-1]) > MIN_DIST_BETWEEN_KEYFRAMES:
                self.keyframes.append(j)
                # add an edge to pose graph to the new keyframe
                bundle_frames = self.frames[self.keyframes[-2]:(self.keyframes[-1]+1)]
                self._add_edge_to_pose_graph(self.keyframes[-2], self.keyframes[-1], self.tracks, bundle_frames)
                # check possible loop closures between new keyframe and previous keyframes:
                if self._check_preconditions_for_loop_closure(j):
                    self.check_loop_closures(sf_j)
        
        # optimize pose graph and output results
        values, marginals = self.pose_graph.optimize()
        self.save_output(values, marginals)

    def check_loop_closures(self, sf_j):
        """ For frame j, go over all frames k before j, attempt loop closure"""
        j = sf_j.idx
        # first, find shortest paths from new keyframe to all previous keyframes
        predecessors = dijkstra(self.distance_matrix.tocsr(), False, j, True)[1]
        for k in self.keyframes[0:-30]:
            # Filter candidates using Mahalanobis distance and PnP inliers percent
            mahalo_distance = compute_mahalanobis_distance(j, k, self.poses, self.marginals, predecessors)
            if mahalo_distance < MAHAL_DISTANCE_THRESHOLD:
                sf_k = self.get_stereo_features(k)
                pose_from_j_to_k, pnp_inliers_matches_k_j, inliers_percent = self._compute_pose_with_pnp(sf_k, sf_j, max_iters=15)
                if inliers_percent > INLIERS_PERCENT_THRESHOLD:
                    self.poses.set_pose_from(j).to(k).with_(pose_from_j_to_k)
                    self.last_loop_closure_idx = j
                    # add loop closure edge to pose graph
                    tracks_k_j = StereoTracksDB().add_frame(sf_k, sf_j, pnp_inliers_matches_k_j)
                    self._add_edge_to_pose_graph(k, j, tracks_k_j, frames=[k, j])
    
    def _add_edge_to_pose_graph(self, keyframe1, keyframe2, tracks, frames):
        # Find pose and covariance between keyframes, using tracks to create a factor graph
        pixel_graph = PixelGraph(self.k, self.ext_l_to_r).build(frames, self.poses, tracks)
        values, marginals = pixel_graph.optimize()
        
        # extract relative pose and covariance from factor graph
        pose_from_2_to_1 = values.atPose3(X(keyframe1)).between(values.atPose3(X(keyframe2)))
        cov_2_conditional_on_1 = g_utils.conditional_covariance_from_Marginals(marginals, keyframe1, keyframe2)
        
        # update poses, covariances and distance matrix with bundle results
        self.poses.set_pose_from(keyframe2).to(keyframe1).with_(pose_from_2_to_1)
        self.marginals.set_cov_of(keyframe2).conditional_on(keyframe1).with_(cov_2_conditional_on_1)
        self.distance_matrix[keyframe2, keyframe1] = np.linalg.det(cov_2_conditional_on_1)
        
        # add factor to pose graph
        self.pose_graph.add_factor(keyframe1, keyframe2, pose_from_2_to_1, cov_2_conditional_on_1)
        return pose_from_2_to_1
    
    def _check_preconditions_for_loop_closure(self, j):
        # only check for loop closures after enough frames have passed, and we haven't added one too recently
        dist_from_last_loop_closure = j - self.last_loop_closure_idx
        return len(self.keyframes) > 30 and dist_from_last_loop_closure > FRAMES_BETWEEN_LOOP_CLOSURES

    def _compute_pose_with_pnp(self, sf_i, sf_j, max_iters=np.inf):
        """ compute pose from camera j to camera i, by matching their stereo-features (keypoints + point-cloud), and calling PnP.

        :param sf_i: StereoFeatures (2D keypoints + point-cloud) of frame i
        :param sf_j: StereoFeatures (2D keypoints + point-cloud) of frame j
        :return:
            pose_from_j_to_i - (4,4)  extrinsics matrix from camera j to camera i
            pnp_inliers_matches_i_j - inliers from matched features between i and j w.r.t. pose_from_j_to_i
            inliers_percent - percent of PnP inliers out of matched pixels between i and j
        """
        # Match keypoints between frame i and frame j
        matches_i_j, sf_i_matched, sf_j_matched = match_two_stereo_pairs(self.features,sf_i, sf_j)
        # Compute PnP using RANSAC
        pose_from_j_to_i, pnp_inliers_bool, inliers_percent = pnp.pnp_stereo_ransac(sf_i_matched, sf_j_matched, self.k, self.ext_l_to_r, max_iters)
        pnp_inliers_matches_i_j = list(compress(matches_i_j, pnp_inliers_bool))
        return pose_from_j_to_i, pnp_inliers_matches_i_j, inliers_percent

    def get_stereo_features(self, idx):
        if idx in self.sf:
            return self.sf[idx]
        left_img, right_img = get_stereo_images(idx, self.dataset_path)
        sf_idx = compute_stereo_features(left_img, right_img, self.features, self.k, self.ext_l_to_r, idx)
        self.sf[idx] = filter_stereo_features(sf_idx)
        return self.sf[idx]

    def save_output(self, values, marginals):
        poses = [values.atPose3(X(i)).matrix() for i in self.keyframes]
        conditional_covariances = []
        for i, j in zip(self.keyframes[:-1], self.keyframes[1:]):
            cov_j_conditional_on_i = g_utils.conditional_covariance_from_Marginals(marginals, i, j)
            conditional_covariances.append(cov_j_conditional_on_i)

        import utils.geometry
        my_dws = utils.geometry.get_dws_from_cam_to_world_s(poses)
        my_sd = (my_dws, 'final', 'red')
        import kitti
        kitti_sd = kitti.get_sd(self.keyframes)
        import utils.plot
        utils.plot.plot_3D_cams([my_sd, kitti_sd], self.keyframes, title="StereoSLAM", plot_dir=self.args.out_dir, save=True, plot=False)
        
        # serialize
        d = {'poses': poses, 'conditional_covariances': conditional_covariances}
        pickle_path = os.path.join(self.args.out_dir, "poses_marginals.pkl")
        with open(pickle_path, 'wb') as handle:
            pickle.dump(d, handle)        
    
  