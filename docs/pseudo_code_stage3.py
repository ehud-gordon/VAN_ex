# graph = NonlinearFactorGraph()
#
# # Add Initial Estimates for Cameras
# initialEstimate = Values()
# initialEstimate.insert(symbol('x',0), Pose3())
#
# kitti_pose_to_world_mat_1 = kitti.read_poses(idx=1)
# initialEstimate.insert(symbol('x',1), kitti_pose_to_world_mat_1)
#
# # Add Factors to Graph
# pose_noise= noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001, 0.1, 0.1, 0.1]))
# meas_noise= noiseModel.Isotropic.Sigma(3, 1.0)
#
# # add Prior Factor for Initial Pose
# graph.add(PriorFactorPose3(symbol('x',0), Pose3(), pose_noise))
#
# # Add Factors for all measurements
# for track in get_tracks(camera_id=0):
#     stereo_pt = StereoPoint2(track.left_x, track.right_x, track.left_y)
#     factor = GenericStereoFactor3D(stereo_point, meas_noise,
#                                    symbol('x', 0), symbol('l', track.id),
#                                    K_matrix))
#     graph.add(factor)
#     # Add initial Estimates for landmarks
#     initialEstimate.insert(symbol('l', track.id), Point3(track.point_cloud))
#
# for track in get_tracks(camera_id=1):
#     if track_is_not_matched_to_frame0:
#         continue
#     stereo_pt = StereoPoint2(track.left_x, track.right_x, track.left_y)
#     factor = GenericStereoFactor3D(stereo_point, meas_noise,
#                                     symbol('x',1), symbol('l', track.id),
#                                     K_matrix))
#     graph.add(factor)