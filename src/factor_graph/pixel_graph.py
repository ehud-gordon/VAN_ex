""" Performs bundle adjustment using matched (stereo) pixles as constraints"""

import numpy as np
from numpy import pi
import gtsam
from gtsam import Pose3, StereoPoint2, GenericStereoFactor3D
from gtsam.symbol_shorthand import X,P
from factor_graph import gtsam_utils

class PixelGraph:
    def __init__(self, k, ext_l_to_r):
        """
        :param k: intrinsics camera matrix in format for use by gtsam GenericStereoFactor3D
        :param ext_l_to_r: extrinsics matrix from left to right stereo camera
        """
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1*pi/180, 1*pi/180, 1*pi/180, 0.3, 0.3, 0.3]))
        self.stereo_pixels_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        self.stereo_cal = gtsam_utils.get_stereo_cal_camera(k, ext_l_to_r)
        
    def build(self, frames, poses, tracks):
        """ builds factor graph using matched stereo pixels as constraints.

        :param frames: list of frames indices
        :param poses: Poses object, allows computation of poses between cameras
        :param tracks: StereoTracksDB object, tracks' database
        """
        startframe, endframe = frames[0], frames[-1]
        self.startframe, self.endframe = startframe, endframe
        self.graph = gtsam.NonlinearFactorGraph()
        self.initialEstimate = gtsam.Values()

        # Add initial estimates for cameras poses
        self.initialEstimate.insert(X(startframe), Pose3())
        for i, j in zip(frames, frames[1:]):
            pose_from_j_to_i = poses.get_pose_from(j).to(i) 
            pose_from_i_to_startframe = self.initialEstimate.atPose3(X(i))
            pose_from_j_to_startframe = pose_from_i_to_startframe.compose(pose_from_j_to_i)
            self.initialEstimate.insert( X(j), pose_from_j_to_startframe )

        # Add Prior Factor for start pose
        # Noise model for prior : 1 deg for cameras rotation, 0.3 meter for 3D location
        priorFactor = gtsam.PriorFactorPose3(X(startframe), Pose3(), self.prior_noise)
        self.graph.add(priorFactor)

        # Add factors for all tracks
        for frame in frames:
            for track in tracks.get_tracks(frame):
                # skip irrelevant tracks
                if (frame==startframe and track.next_track is None) or track.orig_cam_id==endframe:
                    continue

                # add StereoFactor
                stereo_point = StereoPoint2(track.point2_left_x, track.point2_right_x, track.point2_left_y)
                stereo_factor = GenericStereoFactor3D(stereo_point, self.stereo_pixels_noise, X(frame), P(track.id), self.stereo_cal)
                self.graph.add(stereo_factor)
                
                # if this is the first time we add this track, add initial estimate
                if P(track.id) not in self.initialEstimate.keys():
                    # convert point to startframe CS 
                    pose_from_frame_to_startframe = self.initialEstimate.atPose3(X(frame))
                    point3_in_startframe_cs = pose_from_frame_to_startframe.transformFrom(track.point3)
                    self.initialEstimate.insert( P(track.id), point3_in_startframe_cs)

        return self

    def optimize(self):
        """
        :return:
            values - gtsam.Values, containing estimations of landmarks and camera locations
            marginals - gtsam.Marginals, allows extraction of conditional covariance
        """
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph , self.initialEstimate)
        values = optimizer.optimize()
        # extract factor errors for
        marginals = gtsam.Marginals(self.graph, values)
        return values, marginals