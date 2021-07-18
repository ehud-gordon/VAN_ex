"""
Performs SLAM on a set of stereo images, finds cameras locations using a combination of PnP, Bundle Adjustment,
Pose Graph and Loop Closure.
"""
import pickle

import gtsam
from gtsam import Pose3
from gtsam.symbol_shorthand import X
import numpy as np
from numpy import pi
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import lil_matrix
import os
from itertools import compress
from collections import defaultdict
import utils
import utils.plot

from pgm import tracks
from pgm.poses import Poses
from pgm.marginals import Marginals
from features.features2d import Features, filter_with_matches
from pgm import my_bundle
from slam_utils import *
from calib3d import pnp
from calib3d.triangulate import triangulate
import pgm.utils as g_utils
from utils.shortest_path import shortest_path

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.5g" % x))


MIN_DIST_BETWEEN_KEYFRAMES = 5 # minimum distance between keyframes, in meters
MAHAL_DIST_THRESH = 0.4 # threshold for Mahalanobis distance to use in loop closure
INLIERS_PERCENT_THRESH = 0.6 # threshold for percent of inliers for loop closure

class StereoSLAM:
    def __init__(self, args, k, ext_l_to_r):
        """
        :param k: the intrinsics matrix, shared by both stereo cameras
        :param ext_l_to_r: the extrinsics matrix from left camera to right camera
        """
        self.args = args
        self.k = k # the intrinsics matrix
        self.ext_l_to_r = ext_l_to_r # extrinsic left to right
        self.poses = Poses() # poses between cameras
        self.marginals = Marginals() # conditional covariances between two poses
        self.startframe = args.startframe
        self.dataset_path = args.dataset_path
        self.num_frames = len(args.frames)
        self.ext_id = np.diag([1,1,1,1]) # the (4,4) identity matrix, useful as extrinsics matrix
        self.gt_k = g_utils.get_gt_k(k, ext_l_to_r) # camera matrix in format required for gtsam stereo
        self.frames = args.frames # list of frames indices
        self.distance_matrix = lil_matrix((self.num_frames,self.num_frames), dtype=float)
        self.sf = dict() # StereoFeatures (keypoints + point-cloud)
        self.num_of_lc = 0 # TODO del
        self.sd_list = [] # TODO del

    def main(self):
        args = self.args
        self.features = Features(**vars(args))  # used for keypoints detection, description and matching
        self.tracks_db = tracks.StereoTracksDB()
        self.keyframes = [args.startframe]
        last_keyframe = self.keyframes[-1]
        last_keyframe_idx = 0 # TODO del?
        last_loop_closure_idx = 0
        
        self.bundle_li_to_l0_s = [self.ext_id]
        self.pose_graph = gtsam.NonlinearFactorGraph()
        # add prior factor to graph
        pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1*pi/180, 1*pi/180, 1*pi/180, 0.3, 0.3, 0.3])  )  # std, (1 deg, input must be radian), 0.3
        priorFactor = gtsam.PriorFactorPose3(X(self.keyframes[0]), Pose3(), pose_noise_model)
        self.pose_graph.add(priorFactor)
        self.initialEstimate = gtsam.Values()
        self.initialEstimate.insert(X(self.keyframes[0]), Pose3())
    ## Iterate over all frames, find poses using PnP, and optimize with bundle adjustment, pose graph and loop closure
        # i and j denote two consecutive frames/cameras, with i before j
        for j_idx in range(1, self.num_frames):
            j = self.frames[j_idx]
            i = self.frames[j_idx-1]
        # for j in self.frames[1:]:
            # i = self.frames[j-1] 
        ## Compute pose between frames i and j using PnP, get sf (stereo features) for them, and their inliers
            ext_cam_j_to_cam_i, sf_i, sf_j, pnp_inliers_matches_i_j,_ = self._compute_pose_with_pnp(i, j)
            self.poses.set_pose_from(j).to(i).with_(ext_cam_j_to_cam_i)
            # Add PnP inliers to tracks databate, for use in bundle
            self.tracks_db.add_frame(sf_i, sf_j, pnp_inliers_matches_i_j)
            del self.sf[i] # TODO
            if self.poses.get_distance_from(j).to(last_keyframe) < MIN_DIST_BETWEEN_KEYFRAMES:
                continue

        ## Add keyframe: Extract pose and covariance to new keyframe, using a factor graph
            # Perform bundle adjustment on factor graph,  with tracks as constraints
            bundle_frames = self.frames[last_keyframe_idx:(j_idx+1)]
            print(f"added keyframe at j={j}, # {len(self.keyframes)}")
            values, marginals, _,_ = my_bundle.do_single_bundle(bundle_frames, self.poses, self.tracks_db, self.gt_k)
            # TODO del
            ext_lj_to_li = values.atPose3(X(last_keyframe)).between(values.atPose3(X(j))).matrix()
            ext_lj_to_l0 = self.bundle_li_to_l0_s[-1] @ ext_lj_to_li
            self.bundle_li_to_l0_s.append( ext_lj_to_l0 )
            # Extract results from bundle
            pose_from_j_to_keyframe = values.atPose3(X(last_keyframe)).between(values.atPose3(X(j)))
            cov_j_conditional_on_last_keyframe = g_utils.extract_conditional_covariance(marginals, last_keyframe, j)
            # update poses and covariances with bundle results
            self.poses.update_with_Values(bundle_frames, values)
            self.marginals.set_cov_of(j).conditional_on(last_keyframe).with_(cov_j_conditional_on_last_keyframe)
            # add edge to distance matrix with weight of determinant of conditional covariance
            self.distance_matrix[j,last_keyframe] = np.linalg.det(cov_j_conditional_on_last_keyframe)
            # Add factor to pose graph
            # TODO or maybe
            # cov_j_conditionla_on_world = self.marginals.get_cov_of(j).conditional_on(0)
            # noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_j_conditionla_on_world)
            noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_j_conditional_on_last_keyframe)
            factor = gtsam.BetweenFactorPose3(X(last_keyframe), X(j), pose_from_j_to_keyframe, noise_model)
            self.pose_graph.add(factor)
            self.initialEstimate.insert(X(j), Pose3(ext_lj_to_l0))

            # update keyframes
            self.keyframes.append(j)
            last_keyframe = self.keyframes[-1]
            last_keyframe_idx = j_idx # TODO del

            if len(self.keyframes) < 30 or j - last_loop_closure_idx < args.num_frames_between_lc:
                continue

        ## Add possible loop closures between new keyframe and previous keyframes
            # find shortest paths from new keyframe to all previous keyframes
            # noinspection PyTupleAssignmentBalance # TODO del
            _, predecessors = dijkstra(self.distance_matrix.tocsr(), directed=False, indices=last_keyframe, return_predecessors=True)
            added_edges = False
            for k in self.keyframes[0:-30]:
                # Filter candidates using Mahalanobis distance and PnP inliers percent
                mahal_dist = compute_mahalanobis_distance(j, k, self.poses, self.marginals, predecessors)
                if mahal_dist > MAHAL_DIST_THRESH:
                    continue
                # Attempt Consensus match between frames j and k
                ext_cam_j_to_cam_k, sf_k, sf_j, pnp_inliers_matches_k_j, inliers_percent = self._compute_pose_with_pnp(k,j, max_iters=15)
                if inliers_percent < args.inliers_percent:
                    continue
                print(f"added edge between n={j} and i={k}, mahal_dist={mahal_dist}, inliers_perc={inliers_percent}")
                last_loop_closure_idx = j
                # else, close a loop
                self.poses.set_pose_from(j).to(k).with_(ext_cam_j_to_cam_k)
                self._loop_closure(sf_k, sf_j, pnp_inliers_matches_k_j)
                added_edges = True

            if added_edges:
                self._pose_graph()
                self.num_of_lc += 1
                self.output_results()

        self.serialize(title="final", tracks=True)

    def _compute_pose_with_pnp(self, i, j, max_iters=np.inf):
        """ compute pose from camera j to camera i, by computing their stereo-features (keypoints + point-cloud),
        matching them, and calling PnP.

        :param i: index of frame i
        :param j: index of frame j
        :return:
            ext_cam_j_to_cam_i - (4,4)  extrinsic matrix from camera j to camera i
            sf_i - StereoFeatures (2D keypoints + point-cloud)  of frame i
            sf_j - StereoFeatures (2D keypoints + point-cloud)  of frame j
            pnp_inliers_matches_i_j - inliers from matched features between i and j w.r.t. ext_cam_j_to_cam_i
            inliers_percent - percent of PnP inliers out of matched pixels between i and j
        """
        features = self.features
        ## Find keypoints and point-cloud in frame i
        if i not in self.sf:
            self.sf[i] = compute_stereo_features(i, features, self.dataset_path, self.k, self.ext_l_to_r)

        ## Find keypoints and point-cloud in frame j
        if j not in self.sf:
            self.sf[j] = compute_stereo_features(j, features, self.dataset_path, self.k, self.ext_l_to_r)

        ## Match keypoints between frame i and frame j
        matches_i_j, kp_left_i_matched, kp_right_i_matched, pc_j_matched = match_two_stereo_pairs(features, self.sf[i], self.sf[j])
        ext_cam_j_to_cam_i, pnp_inliers_bool, inliers_percent = pnp.pnp_stereo_ransac(kp_left_i_matched, kp_right_i_matched,
                                                                     pc_j_matched, self.k, self.ext_l_to_r, max_iters)
        pnp_inliers_matches_i_j = list(compress(matches_i_j, pnp_inliers_bool))
        return ext_cam_j_to_cam_i, self.sf[i], self.sf[j], pnp_inliers_matches_i_j, inliers_percent

    def _loop_closure(self, sf_k, sf_j, pnp_inliers_matches_k_j):
        """ Perform small bundle adjustment between frames j and k, and updates distance matrix.

        :param sf_k: StereoFeatures object (keypoints + point-cloud) of frame k
        :param sf_j: StereoFeatures object (keypoints + point-cloud) of frame j
        :param pnp_inliers_matches_k_j: inliers from computing PnP between j and k
        """
        j, k = sf_j.idx, sf_k.idx
        tracks_db_k_j = tracks.StereoTracksDB()
        tracks_db_k_j.add_frame(sf_k, sf_j, pnp_inliers_matches_k_j)
        values, marginals, error_before, error_after = my_bundle.do_single_bundle([k,j], self.poses, tracks_db_k_j, self.gt_k)
        pose_from_j_to_k = values.atPose3(X(k)).between(values.atPose3(X(j)))
        print(f'bundlon between j={j},k={k}, error_before={error_before}, error_after={error_after}') # TODO del
        # update poses and covariances with results of bndle
        self.poses.set_pose_from(j).to(k).with_(pose_from_j_to_k)
        cov_j_conditional_on_k = g_utils.extract_conditional_covariance(marginals,k,j)
        self.marginals.set_cov_of(j).conditional_on(k).with_(cov_j_conditional_on_k)
        # add edge to distance matrix
        self.distance_matrix[j,k] = np.linalg.det(cov_j_conditional_on_k)
        # update pose graph
        noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_j_conditional_on_k)
        # TODO or maybe
        # cov_j_conditional_on_world = self.marginals.get_cov_of(j).conditional_on(0)
        # noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_j_conditional_on_world)
        factor = gtsam.BetweenFactorPose3( X(k), X(j), pose_from_j_to_k, noise_model)
        self.pose_graph.add(factor)

    def _pose_graph(self):
        # TODO we only need to call optimize, this entire function too big
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.pose_graph, self.initialEstimate)
        error_before = optimizer.error()
        values = optimizer.optimize()
        error_after = optimizer.error()
        print(f'pose_graph(), error before {error_before}, error after {error_after}')
        marginals = gtsam.Marginals(self.pose_graph, values)
        # update after pose graph
        self.pose_li_to_l0_s = [self.ext_id]
        for j in self.keyframes[1:]:
            self.pose_li_to_l0_s.append(  values.atPose3(X(j)).matrix()  )
        return values, marginals, error_before, error_after


    
    def serialize(self, title="", tracks=False):
        new_poses = defaultdict(dict)
        for j in self.poses.poses.keys():
            for i, pose3 in self.poses.poses[j].items():
                new_poses[j][i] = pose3.matrix()
        d = dict()
        if tracks:
            d['tracks_db'] = self.tracks_db
        d['poses'] = new_poses
        d['marginals'] = self.marginals
        d['bundle_li_to_l0_s'] = self.bundle_li_to_l0_s
        d['pose_li_to_l0_s'] = self.pose_li_to_l0_s
        d['keyframes'] = self.keyframes
        d['dist_matrix'] = self.distance_matrix
        pkl_path = os.path.join(self.args.out_path, f"{title}_poses_marginals.pkl")
        with open(pkl_path, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return pkl_path

    def output_results(self):
        title = f"{self.args.run_name}_lc_{self.num_of_lc}"
        self.serialize(title=title)
        bundle_dws = utils.geometry.get_dws_from_cam_to_world_s(self.bundle_li_to_l0_s)
        kitti_dws = kitti.read_dws(self.keyframes)
        pose_dws = utils.geometry.get_dws_from_cam_to_world_s(self.pose_li_to_l0_s)
        kitti_sd = (kitti_dws, 'kitti', 'green')
        bundle_sd = (bundle_dws, 'Stage 3 (bundle)', 'red') 
        pose_lc_sd = (pose_dws, f'After LC {self.num_of_lc}', utils.plot.get_color(self.num_of_lc))
        self.sd_list.append(pose_lc_sd)
        sd_list = [kitti_sd, bundle_sd] + self.sd_list
        utils.plot.plotly_3D_cams(sd_list, title=title, plot_dir=self.args.out_path, frames_idx=self.keyframes, plot=True, save=True, add_sliders=False)
        

