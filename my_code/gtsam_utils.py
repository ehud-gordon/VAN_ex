import pickle

import gtsam
from gtsam.symbol_shorthand import X, P
import numpy as np
import matplotlib.pyplot as plt
import utils

import os

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.4g" % x))
### VALUES ####
def get_dws_from_gtsam_values(values):
    Pose3_values = gtsam.utilities.allPose3s(values)
    cam_to_world_poses = [values.atPose3(k) for k in Pose3_values.keys()]
    cam_to_world_trans = [pose.translation() for pose in cam_to_world_poses]
    cam_to_world_trans = np.array(cam_to_world_trans).T
    return cam_to_world_trans

def get_points_from_gtsam_values(values):
    points = []
    for k in values.keys():
        try:
            p =values.atPoint3(k)
            points.append(p)
        except RuntimeError:
            continue
    points = np.array(points).T
    return points

def get_cam_to_world_ext_mats_from_values(values):
    Pose3_values = gtsam.utilities.allPose3s(values)
    cam_to_world_poses = [values.atPose3(k) for k in Pose3_values.keys()]
    cam_to_world_mats = [pose.matrix() for pose in cam_to_world_poses]
    return cam_to_world_mats

def get_Pose3_values_from_cam_to_world_ext_mats(ext_mats, frames_idx):
    assert len(ext_mats) == len(frames_idx)
    Pose3_values = gtsam.Values()
    for mat, frame_idx in zip(ext_mats, frames_idx):
        cur_Pose3 = gtsam.Pose3(mat)
        Pose3_values.insert(X(frame_idx), cur_Pose3)
    return Pose3_values

def get_world_to_cam_rot_trans_from_values(values):
    Pose3_values = gtsam.utilities.allPose3s(values)
    cam_to_world_poses = [values.atPose3(k) for k in Pose3_values.keys()]
    world_to_cam_poses = [pose.inverse() for pose in cam_to_world_poses]
    world_to_cam_rots = np.array([pose.rotation().matrix() for pose in world_to_cam_poses])
    world_to_cam_trans_vecs = [pose.translation() for pose in world_to_cam_poses]
    world_to_cam_trans_vecs = np.array(world_to_cam_trans_vecs).T
    return world_to_cam_rots, world_to_cam_trans_vecs

def get_world_to_cam_ext_from_values(values):
    Pose3_values = gtsam.utilities.allPose3s(values)
    cam_to_world_poses = [values.atPose3(k) for k in Pose3_values.keys()]
    world_to_cam_poses = [pose.inverse() for pose in cam_to_world_poses]
    world_to_cam_exts = [pose.matrix() for pose in world_to_cam_poses]
    return world_to_cam_exts

def r0_to_r1_s_t0_to_t1_s_from_values(values):
    world_to_cam_mats = get_world_to_cam_ext_from_values(values)
    r0_to_r1_s, t0_to_t1_s = utils.r0_to_r1_s_t0_to_t1_s(ext_mats=world_to_cam_mats)
    return r0_to_r1_s, t0_to_t1_s

def serialize_Pose3_values(dir_path, values, frames_idx):
    path = os.path.join(dir_path, 'Pose3_values.pkl')
    utils.make_avail_path(path)
    cam_to_world_mats = get_cam_to_world_ext_mats_from_values(values=values) # ndarray
    assert len(cam_to_world_mats) == len(frames_idx)
    d = dict()
    d['cam_to_world_mats'] = cam_to_world_mats
    d['frames_idx'] = frames_idx
    with open(path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path

def unserialize_Pose3_values(path):
    with open(path, 'rb') as handle:
        d = pickle.load(handle)
    cam_to_world_mats = d['cam_to_world_mats']
    frames_idx = d['frames_idx']
    Pose3_values = get_Pose3_values_from_cam_to_world_ext_mats(cam_to_world_mats, frames_idx)
    return Pose3_values

#### VISUALIZATION ######
def plot_2d_cams_points_from_gtsam_values(values, plot_dir, endframe, startframe=0):
    dws = get_dws_from_gtsam_values(values)
    landmarks = get_points_from_gtsam_values(values)
    plt.figure()
    plt.scatter(x=dws[0], y=dws[2], color="red", marker=(5,2), label="camera")
    plt.scatter(x=landmarks[0], y=landmarks[2], color="blue", label="landmark", alpha=0.2)
    plt.xlabel('x');plt.ylabel('z')
    plt.title(f"2D cameras and landmarks for keyframes [{startframe}-{endframe}]")
    plt.legend()
    path = os.path.join(plot_dir, f'2d_cams_points_{startframe}_{endframe}' + '.png')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
