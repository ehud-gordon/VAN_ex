""" Utility functions to read kitti images and matrices """

import cv2
import numpy as np

import os
import re
import glob

import utils.geometry
from core.pose_vector import PoseVector

def data_path():
    cwd = os.getcwd()
    van_ind = cwd.rfind('VAN_ex')
    base_path = cwd[:van_ind+len('VAN_ex')]
    dataset_path = os.path.join(base_path, 'data', 'dataset05')
    return dataset_path

def get_seq_length(dataset_path=None):
    if dataset_path is None:
        dataset_path = data_path()
    images_path = get_images_path_from_dataset_path(dataset_path)
    image_0_path = os.path.join(images_path, 'image_0')
    seq_length = len(glob.glob(image_0_path+os.sep+'[0-9][0-9][0-9][0-9][0-9][0-9].png'))
    return seq_length

def read_images(idx, dataset_path=None, color_mode=cv2.IMREAD_GRAYSCALE):
    """ read a pair of stereo iamges in dataset_path

    :param color_mode: IMREAD_GRAYSCALE or IMREAD_COLOR
    :return: two images in BGR
    """
    if dataset_path is None:
        dataset_path = data_path()
    images_path = get_images_path_from_dataset_path(dataset_path)
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(os.path.join(images_path, 'image_0', img_name), color_mode)  # left image
    img2 = cv2.imread(os.path.join(images_path, 'image_1', img_name), color_mode)  # right image
    return img1, img2

def read_cameras(dataset_path=None):
    """:return: k - (3,3) intrinsics camera matrix (shared by both stereo cameras).
             m1 - (3,4) extrinstic camera matrix [R|t] of left camera. With kitti this is the identity.
             m2 - (3,4), extrinstic camera matrix [R|t] of right camera.
             to get camera projection matrix from WORLD CS to left image, compute k @ m1  """
    # read camera matrices
    if dataset_path is None:
        dataset_path = data_path()
    images_path = get_images_path_from_dataset_path(dataset_path)
    with open(os.path.join(images_path, 'calib.txt')) as f:
        l1 = f.readline().split()[1:]
        l2 = f.readline().split()[1:]
    p1 = np.array([float(i) for i in l1]).reshape(3,4)
    p2 = np.array([float(i) for i in l2]).reshape(3,4)
    k = p1[:, :3]
    m1 = np.linalg.inv(k) @ p1 # with kitti, this is diag(1,1,1)
    m2 = np.linalg.inv(k) @ p2 # this is l_to_r
    k = np.hstack((k, np.zeros((3,1)))) # (3,4)
    m1 = np.vstack((m1, np.array([0,0,0,1]))) # (4,4)
    m2 = np.vstack((m2, np.array([0, 0, 0, 1]))) # (4,4)
    return k, m1, m2

def read_poses_cam_to_world(idx=None, dataset_path=None):
    """ returns extrinsics camera matrices [R|t], from CAMERA to WORLD.

    :param idx: list of lines to take. If None, reads all poses
    :return: list of (4,4) extrinsics matrices at idx [idx[0]_to_world, ... idx[n]_to_world]
    """
    if idx is not None:
        assert sorted(idx) == idx
    if dataset_path is None:
        dataset_path = data_path()
    poses_path = get_poses_path_from_dataset_path(dataset_path)
    matrices = []
    with open (poses_path) as f:
        if not idx:
            for line in f:
                line = line.rstrip().split()
                m = np.array([float(i) for i in line]).reshape(3, 4)
                m = np.vstack((m, np.array([0,0,0,1])))
                matrices.append(m)
        else:
            for line_num, line in enumerate(f):
                if line_num==idx[0]:
                    line = line.rstrip().split()
                    m = np.array([float(i) for i in line]).reshape(3, 4)
                    m = np.vstack((m, np.array([0, 0, 0, 1])))
                    matrices.append(m)
                    idx = idx[1:]
                    if not idx:
                        return PoseVector(matrices)
    return PoseVector(matrices)

def read_relative_poses_cam_to_world(idx=None, dataset_path=None):
    """ return kitti relative poses between a list of indices

    :param idx: list of indices. If None returns all
    :return: if idx=[0,5,7], returns [0_to_0, 5_to_0, 7_to_5]
    """
    poses = read_poses_cam_to_world(idx, dataset_path).as_np()
    relative_poses = [np.diag([1,1,1,1])]
    pose_i_to_world = poses[0]
    for pose_j_to_world in poses[1:]:
        pose_j_to_i = utils.geometry.B_to_A_mat(pose_i_to_world, pose_j_to_world)
        relative_poses.append(pose_j_to_i)
    return PoseVector(relative_poses)

def read_dws(idx=None, dataset_path=None):
    """ returns dws from kitti original matrices
    :param idx: list (size n) of lines to read. If none read all
    :return: ndarray (3,n)
    """
    matrices = read_poses_cam_to_world(idx=idx, dataset_path=dataset_path).as_np()
    dws = [m[0:3,-1] for m in matrices]
    return np.array(dws).T

def get_sd(idx=None, dataset_path=None):
    dws = read_dws(idx, dataset_path)
    return (dws, 'kitti', 'green')

def get_images_path_from_dataset_path(dataset_path):
    base = os.path.basename(dataset_path) # dataset05
    seq_num = re.search(r"[0-9]+", base).group(0) # 05
    images_path = os.path.join(dataset_path, 'sequences', str(seq_num))
    return images_path

def get_poses_path_from_dataset_path(dataset_path):
    base = os.path.basename(dataset_path) # dataset05
    seq_num = str(re.search(r"[0-9]+", base).group(0)) # 05
    poses_path = os.path.join(dataset_path, 'poses', seq_num+'.txt')
    return poses_path