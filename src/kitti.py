import cv2
import os
import numpy as np
import re
import glob

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
    """
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
    """ read original poses.
    returns extrinsics camera matrices [R|t], with this being
    from CAMERA to WORLD.
    :param idx: list of lines to take
    :return: list of extrinsics matrices [ndarray(4,4), ..., ndarray(4,4)]
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
                        return matrices
    return matrices

def read_dws(idx=None, dataset_path=None):
    """ returns dws from kitti original matrices
    :param idx: list (size n) of lines to read. If none read all
    :return: ndarray (3,n)
    """
    matrices = read_poses_cam_to_world(idx=idx, dataset_path=dataset_path)
    dws = []
    for m in matrices:
        dws.append(m[0:3,-1])
    return np.array(dws).T
