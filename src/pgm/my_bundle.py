""" Performs bundle adjustment using tracks (matched pixles) as constraints"""

import numpy as np
from numpy import pi
import gtsam
from gtsam import Pose3, StereoPoint2, GenericStereoFactor3D, Point3
from gtsam.symbol_shorthand import X, P


def build_bundle_graph(frames_idx, poses, tracks_db, gt_k):
    """ builds Factor Graph using factors of constraints between tracks.

    :param frames_idx: list of frames indices
    :param poses: Poses object, allows computation of poses between cameras
    :param tracks_db: StereoTracksDB object, tracks' database
    :param gt_k: intrinsics camera matrix in format for use by gtsam GenericStereoFactor3D
    :return:
        graph - Factor graph with factors
        initialEstimate - initial estimate of camera poses and landmarks locations
    """
    startframe, endframe = frames_idx[0], frames_idx[-1]   
    graph = gtsam.NonlinearFactorGraph()
    initialEstimate = gtsam.Values()
    
    # (1 deg, input must be radian), 0.3 meter
    pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1 * pi / 180, 1 * pi / 180, 1 * pi / 180, 0.3, 0.3, 0.3]))
    meas_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

    ## Add initial estimates for cameras poses
    for i in frames_idx:
        initialEstimate.insert( X(i), poses.get_pose_from(i).to(startframe) )

    ## Add Prior Factor for start pose
    priorFactor = gtsam.PriorFactorPose3(X(startframe), Pose3(), pose_noise_model)
    graph.add(priorFactor)

    ## Add factors for all tracks
    for frame in frames_idx:
        for track in tracks_db.get_tracks(frame):
            # filter irrelevant tracks 
            if (frame==startframe and track.next_track is None) or track.orig_cam_id==endframe:
                continue
            
            # add Factor for measurement
            stereo_point = StereoPoint2(track.point2_left_x, track.point2_right_x, track.point2_left_y)
            stereoFactor = GenericStereoFactor3D(stereo_point, meas_noise_model, X(frame), P(track.id), gt_k)
            graph.add(stereoFactor)
            
            # if new point in bundle, add initial estimate
            if frame == max(track.orig_cam_id, startframe):
                # convert point to startframe CS
                pose_from_frame_to_startframe = poses.get_pose_from(frame).to(startframe)
                # get 3d location in startframe coordinate system
                point3_in_startframe_cs = pose_from_frame_to_startframe.transformFrom(track.point3)
                initialEstimate.insert( P(track.id), point3_in_startframe_cs )
    return graph, initialEstimate

def do_single_bundle(frames_idx, poses, tracks_db, gt_k):
    """ Perform bundle adjustment with tracks as constraints. Builds Factor Graph and optizimies it.

    :param frames_idx: list of frames indices
    :param poses: Poses object, allows computation of poses between cameras
    :param tracks_db: StereoTracksDB object, tracks' database
    :param gt_k: intrinsics camera matrix in format for use by gtsam GenericStereoFactor3D
    :return:
        values - gtsam.Values object, containing estimations of camera locations
        marginals - gtsam.Marginals, allows extraction of conditional covariance
        error_before - Factor Graph error before optimization
        error_after - Factor Graph error after optimization
    """
    # build graph
    graph, initialEstimate = build_bundle_graph(frames_idx, poses, tracks_db, gt_k)
    # optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
    error_before = optimizer.error()
    values = optimizer.optimize()
    error_after = optimizer.error()
    marginals = gtsam.Marginals(graph, values)
    return values, marginals, error_before, error_after
