""" Utility methods for computer vision geometry """

import cv2
import numpy as np

def rodrigues_to_mat(rvec,tvec):
    """ create extrinsics matrix from a rotation and a translation vectors.

    :param rvec: (3,) rotation vector
    :param tvec: (3,) translation vector
    :return: (4,4) extrinsics matrix
    """
    rot = cv2.Rodrigues(src=rvec)[0] # (3,3)
    extrinsic = np.hstack((rot, tvec)) # (3,4)
    extrinsic = np.vstack((extrinsic, np.array([0,0,0,1])))
    return extrinsic # (4,4)

def transform_pc_to_world(pose_cam_to_world, pc):
    """ project point cloud (in camera CS) to world CS

    :param pose_cam_to_world (4,4) or (3,4) extrinsic matrix
    :param pc: ndarray (3,n) point-cloud in camera coordinate-system
    :return: ndarray (3,n) point-cloud in world coordinate-system
    """
    res = (pose_cam_to_world[:3, :3] @ pc[:3]) + (pose_cam_to_world[:3, -1][:, None])
    return res

def get_dws_from_cam_to_world_s(cam_to_world_mats):
    """ extract translation vectors from a set of extrinsics matrices

    :param cam_to_world_mats:  list of size n
    :return: (3,n) ndarray
    """
    res = [mat[0:3,3] for mat in cam_to_world_mats]
    return np.array(res).T

def inv_extrinsics(ext_mat):
    """ inverse transformation
    :param ext_mat: (3,4) or (4,4) extrinsics matrix from CS A to B
    :return: the matrix (3,4) or (4,4) from CS B to A
    """
    assert ext_mat.shape in [(3, 4), (4, 4)]
    r,t = get_rot_trans(ext_mat)
    inv = np.hstack((r.T, (r.T@-t).reshape(3,1)))
    if ext_mat.shape == (4,4):
        inv = np.vstack((inv, np.array([0,0,0,1])))
    return inv

def rotation_matrices_diff(R,Q):
    rvec, _ = cv2.Rodrigues(R.transpose() @ Q)
    radian_diff = np.linalg.norm(rvec)
    deg_diff = radian_diff * 180 / np.pi
    return deg_diff

def compare_ext_mats(ext_mat_1, ext_mat_2):
    rot1, trans1 = get_rot_trans(ext_mat_1)
    rot2, trans2 = get_rot_trans(ext_mat_2)
    rot_diff_in_deg = rotation_matrices_diff(rot1, rot2) # in degrees
    trans_diff = np.linalg.norm(trans1-trans2) # L2 norm
    return rot_diff_in_deg, trans_diff

def rot_trans_A_to_B(ext_A_to_world, ext_B_to_world): # between
    rot_A_to_world, trans_A_to_world = get_rot_trans(ext_A_to_world)
    rot_B_to_world, trans_B_to_world = get_rot_trans(ext_B_to_world)
    rot_A_to_B = rot_B_to_world.T @ rot_A_to_world
    trans_A_to_B = rot_B_to_world.T @ (trans_A_to_world - trans_B_to_world)
    return rot_A_to_B, trans_A_to_B

def rot_trans_B_to_A(ext_A_to_world, ext_B_to_world): # between
    """ this is equivalent to calling rot_trans_A_to_B(ext_B_to_world, ext_A_to_world) """
    rot_A_to_world, trans_A_to_world = get_rot_trans(ext_A_to_world)
    rot_B_to_world, trans_B_to_world = get_rot_trans(ext_B_to_world)
    rot_B_to_A = rot_A_to_world.T @ rot_B_to_world
    trans_B_to_A = rot_A_to_world.T @ (trans_B_to_world - trans_A_to_world)
    return rot_B_to_A, trans_B_to_A

def A_to_B_mat(ext_A_to_world, ext_B_to_world):
    rot, trans = rot_trans_A_to_B(ext_A_to_world, ext_B_to_world)
    return rot_trans_to_ext(rot, trans)

def B_to_A_mat(ext_A_to_world, ext_B_to_world):
    rot, trans = rot_trans_B_to_A(ext_A_to_world, ext_B_to_world)
    return rot_trans_to_ext(rot, trans)

def t2v(rot_mat, trans_vec):
    rot_in_rads = rot_mat_2_euler_angles(rot_mat)
    res = np.concatenate((rot_in_rads, trans_vec))
    return res

def rot_mat_2_euler_angles(R):
    """ returns x,y,z in radians"""
    sy = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def get_rot_trans(ext_mat):
    """ return rotation matrix and translation vector from an extrinsics matrix

    :param ext_mat: (3,4) or (4,4) extrinsics matrix
    """
    rotation_matrix = ext_mat[0:3, 0:3].astype('float64')
    translation_vector = ext_mat[0:3, 3].astype('float64')
    return rotation_matrix, translation_vector

def rot_trans_to_ext(rotation_mat, translation_vec):
    translation_vec = translation_vec.reshape(3,1)
    ext_mat = np.hstack((rotation_mat, translation_vec))
    ext_mat = np.vstack((ext_mat, [0,0,0,1]))
    return ext_mat.astype('float64')

