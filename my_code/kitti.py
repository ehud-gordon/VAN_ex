import cv2
import os
import numpy as np
import re
import glob
import utils


DATASET_5_PATH = os.path.join('C:','Users', 'godin','Documents','VAN_ex','data','dataset05')
DATASET_5_PATH_WSL = os.path.join(os.sep, 'mnt','c','Users','godin','Documents','VAN_ex','data','dataset05')
IMAGES_05_PATH = os.path.join(DATASET_5_PATH, 'sequences', '05')
POSES_05_TXT = os.path.join(DATASET_5_PATH, 'poses','05.txt')

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

def read_cameras(dataset_path=None):
    """:return: k - intrinsic matrix (shared by both cameras), np.array 3x3
             m1 - extrinstic camera matrix [R|t] of left camera, np.array 3x4
             m2 - extrinstic camera matrix [R|t] of right camera, np.array 3x4
             to get camera matrix (P1), compute k @ m1  """
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
    m1 = np.linalg.inv(k) @ p1
    m2 = np.linalg.inv(k) @ p2
    k = np.hstack((k, np.zeros((3,1)))) # (3,4)
    m1 = np.vstack((m1, np.array([0,0,0,1]))) # (4,4)
    m2 = np.vstack((m2, np.array([0, 0, 0, 1]))) # (4,4)
    return k, m1,m2

def read_poses_orig(idx, dataset_path=None):
    """ returns extrinsics camera matrices [R|t], with this being
    from CAMERA to WORLD.
    :param idx: list of lines to take
    :return: list of extrinsics matrices [ndarray(4,4), ..., ndarray(4,4)]
    """
    assert sorted(idx) == idx
    matrices = []
    if dataset_path is None:
        dataset_path = data_path()
    poses_path = get_poses_path_from_dataset_path(dataset_path)
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

def read_dws(idx, dataset_path=None):
    """ returns dws from kitti original matries
    :param idx: list of lines to take of size n
    :return: ndarray (3,n)
    """
    matrices = read_poses_orig(idx=idx, dataset_path=dataset_path)
    dws = []
    for m in matrices:
        dws.append(m[0:3,-1])
    return np.array(dws).T

def read_poses_world_to_cam(idx, dataset_path=None):
    matrices = read_poses_orig(idx=idx, dataset_path=dataset_path)
    matrices = [utils.inv_extrinsics(m) for m in matrices]
    return matrices

def read_relative_poses_world_to_cam(idx, dataset_path=None):
    """ from l{idx-1} to to l{idx}"""
    ext_mats = read_poses_world_to_cam(idx, dataset_path)
    r0_to_r1_s, t0_to_t1_s = [], []
    for i in range(1, len(ext_mats)):
        r0_to_r1, t0_to_t1 = utils.r0_to_r1_t0_to_t1(l0=ext_mats[i-1], l1=ext_mats[i])
        r0_to_r1_s.append(r0_to_r1)
        t0_to_t1_s.append(t0_to_t1)
    r0_to_r1_s = np.array(r0_to_r1_s)
    t0_to_t1_s = np.array(t0_to_t1_s)
    return r0_to_r1_s, t0_to_t1_s.T


def read_trans_vectors(idx, dataset_path=None):
    """ returns translation vectors from (WORLD-TO-CAM) kitti matries
    :param idx: list of lines to take of size n
    :return: ndarray (3,n)
    """
    matrices = read_poses_world_to_cam(idx=idx, dataset_path=dataset_path)
    trans_vecs = []
    for m in matrices:
        trans_vecs.append(m[0:3,-1])
    return np.array(trans_vecs).T

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

if __name__=="__main__":
    np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.4g" % x))
    idx = [0,10,20]
    rots, trans = read_relative_poses_world_to_cam(idx=idx)
    z=3