import cv2
import numpy as np


def rodrigues_to_mat(rvec,tvec):
    rot, _ = cv2.Rodrigues(src=rvec)
    extrinsic = np.hstack((rot, tvec))
    extrinsic = np.vstack((extrinsic, np.array([0,0,0,1])))
    return extrinsic # (4,4)


def get_dw_from_world_to_cam(ext_mat):
    """
    :param ext_mat: [R|t] extrinsics matrix of some camera
    :return: dw: (3,1) vector in world coordinates of camera origin
    """
    assert ext_mat.shape in [(3,4), (4,4)]
    r,t = ext_mat[:3,:3], ext_mat[0:3,-1]
    return r.T @ -t


def get_dws_from_world_to_cam_s(world_to_cam_mats):
    dws = [get_dw_from_world_to_cam(ext_mat) for ext_mat in world_to_cam_mats]
    return np.array(dws).T


def get_dws_from_cam_to_world_s(cam_to_world_mats):
    res = [mat[0:3,3] for mat in cam_to_world_mats]
    return np.array(res).T


def inv_extrinsics(ext_mat):
    """
    :param ext_mat: [R|t] extrinsics matrix of some camera in global coordinates
    :return: the [R|T]  matrix of global in camera coordinates, same shape as ext_max
    """
    assert ext_mat.shape in [(3, 4), (4, 4)]
    r,t = get_rot_trans(ext_mat)
    inv = np.hstack((r.T, (r.T@-t).reshape(3,1)))
    if ext_mat.shape == (4,4):
        inv = np.vstack((inv, np.array([0,0,0,1])))
    return inv


def inv_extrinsics_mult(ext_mats):
    return [inv_extrinsics(mat) for mat in ext_mats]


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


def rot_trans_A_to_B(ext_A_to_0, ext_B_to_0): # between
    rot_A_to_0, trans_A_to_0 = get_rot_trans(ext_A_to_0)
    rot_B_to_0, trans_B_to_0 = get_rot_trans(ext_B_to_0)
    rot_A_to_B = rot_B_to_0.T @ rot_A_to_0
    trans_A_to_B = rot_B_to_0.T @ (trans_A_to_0 - trans_B_to_0)
    return rot_A_to_B, trans_A_to_B


def rot_trans_B_to_A(ext_A_to_0, ext_B_to_0): # between
    """ this is equivalent to calling rot_trans_A_to_B(ext_B_to_0, ext_A_to_0) """
    rot_A_to_0, trans_A_to_0 = get_rot_trans(ext_A_to_0)
    rot_B_to_0, trans_B_to_0 = get_rot_trans(ext_B_to_0)
    rot_B_to_A = rot_A_to_0.T @ rot_B_to_0
    trans_B_to_A = rot_A_to_0.T @ (trans_B_to_0 - trans_A_to_0)
    return rot_B_to_A, trans_B_to_A


def A_to_B_mat(ext_A_to_0, ext_B_to_0):
    rot, trans = rot_trans_A_to_B(ext_A_to_0, ext_B_to_0)
    return rot_trans_to_ext(rot, trans)


def B_to_A_mat(ext_A_to_0, ext_B_to_0):
    rot, trans = rot_trans_B_to_A(ext_A_to_0, ext_B_to_0)
    return rot_trans_to_ext(rot, trans)


def rot_trans_i_to_n(ext_0_to_i, ext_0_to_n): # between
    ext_i_to_0 = inv_extrinsics(ext_0_to_i)
    ext_n_to_0 = inv_extrinsics(ext_0_to_n)
    return rot_trans_A_to_B(ext_i_to_0, ext_n_to_0)


def rot_trans_j_to_i_s(ext_i_to_0_s):
    rot_j_to_i_s, trans_j_to_i_s = [], []
    for j in range(1, len(ext_i_to_0_s)):
        i = j-1
        ext_j_to_0, ext_i_to_0 = ext_i_to_0_s[j], ext_i_to_0_s[i]
        rot_j_to_i, trans_j_to_i = rot_trans_B_to_A(ext_i_to_0, ext_j_to_0)
        rot_j_to_i_s.append(rot_j_to_i)
        trans_j_to_i_s.append(trans_j_to_i)
    trans_j_to_i_s = np.array(trans_j_to_i_s).T
    return rot_j_to_i_s, trans_j_to_i_s


def make_relative_to_ci(ext_ci_to_c0_s):
    relative_ci_to_c0_s = [np.diag([1,1,1,1])]
    ci_to_c0 = ext_ci_to_c0_s[0]
    for j in range(1,len(ext_ci_to_c0_s)):
        cj_to_c0 = ext_ci_to_c0_s[j]
        cj_to_ci = B_to_A_mat(ci_to_c0, cj_to_c0)
        relative_ci_to_c0_s.append(cj_to_ci)
    return relative_ci_to_c0_s


def get_rot_trans(ext_mat):
    """ return rotation matrix and translation vector from an extrinsics matrix

    :param ext_mat: (3,4) or (4,4) extrinsics matrix
    """
    rotation_matrix = ext_mat[0:3, 0:3].astype('float64')
    translation_vector = ext_mat[0:3, 3].astype('float64')
    return rotation_matrix, translation_vector


def get_rot_trans_s(mats):
    # TODO remove this method
    rot_s, trans_s = [], []
    for mat in mats:
        r,t = get_rot_trans(mat)
        rot_s.append(r); trans_s.append(t)
    return rot_s, np.array(trans_s).T


def rot_trans_to_ext(rotation_mat, translation_vec):
    translation_vec = translation_vec.reshape(3,1)
    ext_mat = np.hstack((rotation_mat, translation_vec))
    ext_mat = np.vstack((ext_mat, [0,0,0,1]))
    return ext_mat.astype('float64')


def rot_trans_stats(rot_diffs, trans_diffs, frames_idx, rel_or_abs):
    startframe = frames_idx[0]
    endframe = frames_idx[-1]
    num_frames = endframe - startframe
    rots_error_sum = np.sum(rot_diffs)
    rot_error_avg = rots_error_sum / num_frames
    tx_error, ty_error, tz_error = np.sum(trans_diffs, axis=1)
    trans_error_sum = + tx_error + ty_error + tz_error
    trans_error_avg = trans_error_sum / num_frames
    stats = [f"avg. {rel_or_abs} rotation error per frame = {rots_error_sum:.1f}/{num_frames} = {rot_error_avg:.2f} deg",
             f"sum of {rel_or_abs} translation errors frames [{startframe},{endframe}] = {tx_error:.1f} + {ty_error:.1f} + {tz_error:.1f} = {trans_error_sum:.1f} meters",
             f"avg. {rel_or_abs} translation error per frame = {trans_error_sum:.1f}/{num_frames} = {trans_error_avg:.2f} meters"]
    return stats, rots_error_sum,  trans_error_sum


def get_rot_trans_diffs_from_mats(exts_gt, *exts_B):
    """
    :param exts_gt: a list of extrinsics matrices that we compare to (ground truth)
    """
    for exts in exts_B:
        assert len(exts_B) == len (exts_B)
    rots_gt, trans_gt = get_rot_trans_s(exts_gt)
    rots_diffs_res, trans_diffs_res = [], []
    for exts in exts_B:
        rots_B, trans_B = get_rot_trans_s(exts)
        rots_diffs, trans_diffs = get_rot_trans_diffs(rots_gt, rots_B, trans_gt, trans_B)
        rots_diffs_res.append(rots_diffs)
        trans_diffs_res.append(trans_diffs)
    return rots_diffs_res, trans_diffs_res


def get_rot_trans_diffs(rots_A, rots_B, trans_vecs_A, trans_vecs_B):
    """
    :param rots: list of rotation matrices
    :param trans_vecs: (3,n)
    """
    rot_diffs = [rotation_matrices_diff(r, q) for r, q in zip (rots_A, rots_B)]
    rot_diffs = np.array(rot_diffs)
    trans_diffs = np.abs(trans_vecs_A - trans_vecs_B)
    return rot_diffs, trans_diffs


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


def rot_trans_norm(rot, trans):
    rot_deg_norm = rotation_matrices_diff(np.diag([1, 1, 1]), rot)
    trans_norm = np.linalg.norm(trans) # in meters
    return rot_deg_norm, trans_norm


def rot_trans_norm_from_ext(ext_mat):
    rot, trans = get_rot_trans(ext_mat)
    return rot_trans_norm(rot, trans)


def rot_trans_norms_from_exts(ext_mats):
    rot_norms, trans_norms = [], []
    for mat in ext_mats:
        r,t = get_rot_trans(mat)
        rot_norm, trans_norm = rot_trans_norm(r,t)
        rot_norms.append(rot_norm)
        trans_norms.append(trans_norm)
    return rot_norms, trans_norms


def t2v(rot_mat, trans_vec):
    rot_in_rads = rot_mat_2_euler_angles(rot_mat)
    res = np.concatenate((rot_in_rads, trans_vec))
    return res


def concat_ci_to_cj_s(ext_ci_to_cj_s):
    ext_c0_to_ci_s = [ext_ci_to_cj_s[0]]
    for j in range(1, len(ext_ci_to_cj_s)): # i=j-1
        ext_ci_to_cj = ext_ci_to_cj_s[j]
        ext_c0_to_cj = ext_ci_to_cj @ ext_c0_to_ci_s[-1]
        ext_c0_to_ci_s.append(ext_c0_to_cj)
    return ext_c0_to_ci_s


def concat_and_inv_ci_to_cj_s(ext_ci_to_cj_s):
    ext_c0_to_ci_s = concat_ci_to_cj_s(ext_ci_to_cj_s)
    ext_ci_to_c0_s = inv_extrinsics_mult(ext_c0_to_ci_s)
    return ext_ci_to_c0_s


def concat_cj_to_ci_s(ext_cj_to_ci_s):
    ext_ci_to_c0_s = [ext_cj_to_ci_s[0]]
    for j in range(1, len(ext_cj_to_ci_s)):  # i=j-1
        ext_cj_to_ci = ext_cj_to_ci_s[j]
        ext_cj_to_c0 = ext_ci_to_c0_s[-1] @ ext_cj_to_ci
        ext_ci_to_c0_s.append(ext_cj_to_c0)
    return ext_ci_to_c0_s