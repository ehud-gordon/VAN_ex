import pickle

import matplotlib.pyplot as plt
import cv2
import numpy as np

import os
from datetime import datetime
import re
import shutil

CYAN_COLOR = (255,255,0) # in BGR
ORANGE_COLOR = (0, 128,255) # in BGR
GREEN_COLOR = (0,255,0) # in BGR
RED_COLOR = (0,0,255) # in BGR
INLIER_COLOR = GREEN_COLOR
OUTLIER_COLOR = RED_COLOR

MATCH_Y_DIST_MAX = 2

#########################   Files   #########################
def dir_name_ext(path):
    folder, base = os.path.split(path)
    if os.path.isdir(path):
        return folder, base,""
    name, ext = os.path.splitext(base)
    return folder, name, ext

def clear_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)

def make_dir_if_needed(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def clear_path(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

def get_avail_path(path):
    while os.path.exists(path):
        folder,name,ext = dir_name_ext(path)
        path = os.path.join(folder, name+'0'+ext)
    return path

def get_time_path():
    return datetime.now().strftime("%m-%d-%H-%M")

def out_dir():
    cwd = os.getcwd()
    van_ind = cwd.rfind('VAN_ex')
    base_path = cwd[:van_ind+len('VAN_ex')]
    res_dir = os.path.join(base_path, 'out')
    return res_dir

def path_to_linux(path):
    parts = re.split(r'\\', path)
    if len(parts) == 1: return path
    right_parts = ['/mnt']
    for p in parts:
        if p=='C:':
            p = 'c'
        
        right_parts.append(p)
    return r'/'.join(right_parts)

def path_to_windows(path):
    parts = re.split(r'/', path)
    if len(parts) == 1: return path
    right_parts = []
    for p in parts[2:]:
        if p=='c':
            p = 'C:'
        right_parts.append(p)
    return '\\'.join(right_parts)

def path_to_current_os(path):
    if os.name == 'nt':
        return path_to_windows(path)
    elif os.name == "posix":
        return path_to_linux(path)
    return path

def serialize_ext_li_to_lj_s(dir_path, ext_li_to_lj_s, title):
    d = {'ext_li_to_lj_s': ext_li_to_lj_s}
    pkl_path = os.path.join(dir_path, f'ext_li_to_lj_s{lund(title)}.pkl')
    clear_path(pkl_path)
    with open(pkl_path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_path

def deserialize_ext_li_to_lj_s(pkl_path):
    with open(pkl_path, 'rb') as handle:
        d = pickle.load(handle)
    return d['ext_li_to_lj_s']

#########################   Geometry   #########################
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
    r,t = get_r_t(ext_mat)
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

def compare_ext_mat(ext_mat_1, ext_mat_2):
    rot1, t1 = get_r_t(ext_mat_1)
    rot2, t2 = get_r_t(ext_mat_2)
    rot_diff_in_deg = rotation_matrices_diff(rot1, rot2) # in degrees
    trans_diff = np.linalg.norm(t1-t2) # L2 norm
    return rot_diff_in_deg, trans_diff

def rot_trans_A_to_B(ext_A_to_0, ext_B_to_0): # between
    rot_A_to_0, trans_A_to_0 = get_r_t(ext_A_to_0)
    rot_B_to_0, trans_B_to_0 = get_r_t(ext_B_to_0)
    rot_A_to_B = rot_B_to_0.T @ rot_A_to_0
    trans_A_to_B = rot_B_to_0.T @ (trans_A_to_0 - trans_B_to_0)
    return rot_A_to_B, trans_A_to_B

def rot_trans_B_to_A(ext_A_to_0, ext_B_to_0): # between
    """ this is equivalent to calling rot_trans_A_to_B(ext_B_to_0, ext_A_to_0) """
    rot_A_to_0, trans_A_to_0 = get_r_t(ext_A_to_0)
    rot_B_to_0, trans_B_to_0 = get_r_t(ext_B_to_0)
    rot_B_to_A = rot_A_to_0.T @ rot_B_to_0
    trans_B_to_A = rot_A_to_0.T @ (trans_B_to_0 - trans_A_to_0)
    return rot_B_to_A, trans_B_to_A

def A_to_B_mat(ext_A_to_0, ext_B_to_0):
    rot, trans = rot_trans_A_to_B(ext_A_to_0, ext_B_to_0)
    return r_t_to_ext(rot, trans)

def B_to_A_mat(ext_A_to_0, ext_B_to_0):
    rot, trans = rot_trans_B_to_A(ext_A_to_0, ext_B_to_0)
    return r_t_to_ext(rot, trans)

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

def get_r_t(mat):
    r = mat[0:3, 0:3]
    t = mat[0:3,3]
    return r.astype('float64'),t.astype('float64')

def get_r_t_s(mats):
    rot_s, trans_s = [], []
    for mat in mats:
        r,t = get_r_t(mat)
        rot_s.append(r); trans_s.append(t)
    return rot_s, np.array(trans_s).T

def r_t_to_ext(r,t):
    mat = np.hstack((r, t.reshape(3,1)))
    mat = np.vstack((mat, [0,0,0,1]))
    return mat.astype('float64')

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

def get_consistent_with_extrinsic(kp_l, kp_r, pc, ext_w_to_c, ext_l_to_r, k, reproj_thresh=2):
    """
    Computes the inliers of kp-pc match w.r.t. extrinsic matrices.
    The inputs are:
    (1) matching pixels in left (kp_l) and right (kp_r) images, we believe with no incorrect matches
    (2) uncertain locations of their 3D location (in world CS),
    (3) Extrinsics matrices: ext world_to_cam and ext_left_to_right that we belive is correct.
    We project pc to pixels_left and pixels_right, using the ext_mats, to get proj_pixels_left and proj_pixels_right.
    We then filter, so we keep only those points that're consistent with the ext_mats. That is, their world-point is projected close
    to both their pixels_left and pixels_right locations.
    :param kp_l/r: (2,n)  matching keypoints in pixels_left / pixels_right
    :param pc: (3,n) points in world CS, matching estimates of kp_l and kp_r real world locations.
    :param ext_w_to_c: (4,4) extrinsic matrix from world to left camera
    :param ext_l_to_r: (4,4) extrinsic matrix from left camera to right camera
    :param k: (4,4) intrinsics camera matrix
    :return inliers_bool - boolean ndarray (n,), with True in indices of points that're consistent
    """
    assert kp_l.shape[1] == kp_r.shape[1] == pc.shape[1]
    if pc.shape[0] == 3:
        pc = np.vstack((pc, np.ones(pc.shape[1])))  # (4,n)

    proj_w_to_l = k @ ext_w_to_c  # (3,4) # from world to pixels_left
    ext_w_to_r = ext_l_to_r @ ext_w_to_c  # (4,4) # from world to camera_right
    proj_w_to_r = k @ ext_w_to_r  # (3,4) # from word to pixels_right

    # project pc to pixels_left
    projected_l = proj_w_to_l @ pc  # (3,n)
    projected_l = projected_l[0:2] / projected_l[-1]  # (2,n)

    # project pc to pixels_right
    projected_r = proj_w_to_r @ pc  # (3,n)
    projected_r = projected_r[0:2] / projected_r[-1]  # (2,n)

    proj_errors_l = np.linalg.norm((kp_l - projected_l), axis=0) # L2 norm
    proj_errors_r = np.linalg.norm((kp_r - projected_r), axis=0)  # L2 norm
    bool_l = proj_errors_l <= reproj_thresh
    bool_r = proj_errors_r <= reproj_thresh
    inliers_bool = bool_l * bool_r # (n,)
    return inliers_bool, proj_errors_l, proj_errors_r

def get_rot_trans_diffs_from_mats(exts_A, exts_B):
    rots_A, trans_A = get_r_t_s(exts_A)
    rots_B, trans_B = get_r_t_s(exts_B)
    return get_rot_trans_diffs(rots_A, rots_B, trans_A, trans_B)

def get_rot_trans_diffs(rots_A, rots_B, trans_vecs_A, trans_vecs_B):
    """
    :param rots: list of rotation matrices 
    :param trans_vecs: (3,n) 
    """
    rot_diffs = [rotation_matrices_diff(r, q) for r,q in zip (rots_A, rots_B)]
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
    rot_deg_norm = rotation_matrices_diff(np.diag([1,1,1]), rot)
    trans_norm = np.linalg.norm(trans) # in meters
    return rot_deg_norm, trans_norm

def rot_trans_norm_from_ext(ext_mat):
    rot, trans = get_r_t(ext_mat)
    return rot_trans_norm(rot, trans)

def rot_trans_norms_from_exts(ext_mats):
    rot_norms, trans_norms = [], []
    for mat in ext_mats:
        r,t = get_r_t(mat)
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

######################### NUMPY #########################
def cumsum_mats(list_of_mats):
    cumsum_arr = np.cumsum(list_of_mats, axis=0)
    cumsum_list = np.split(cumsum_arr, len(list_of_mats), axis=0)
    cumsum_list = [np.squeeze(mat) for mat in cumsum_list]
    return cumsum_list

def filt_np(bool_array, *nd_arrays):
    """
    :param bool_array: boolean array of size n
    :param nd_arrays: ndarray of size (?,n)
    :return: the arrays filtered
    """
    for arr in nd_arrays:
        assert len(bool_array) == arr.shape[1]
    return [arr[:,bool_array] for arr in nd_arrays]

######################### OTHERS ##################################
def get_perc_largest_indices(arr, perc):
    """
    find indices of the percent largest element in array
    :param arr: (n,) ndarray of numbers
    :param perc: number between [0-1], percentage of largest
    :return: boolean array, with True in place of largest perc
    """
    arr_abs = np.abs(arr)
    size = arr_abs.size
    num_of_largest = int(size * perc)
    idx_of_largest = np.argpartition(-arr_abs, num_of_largest)
    bool_array = np.zeros_like(arr, dtype=bool)
    bool_array[idx_of_largest[0:num_of_largest]] = True

    return bool_array

def und_title(string):
    return ('_'+string+'_') if string else ""

def lund(string):
    return f'_{string}' if string else ""

def rund(string):
    return f'{string}_' if string else ""

def get_color(i):
    c = ['black', 'cyan', 'orange', 'purple','brown','silver','gold', 'indigo', 'tomato', 'seashell', 'rosybrown', 'plum', 'limegreen']
    return c[i % len(c)]

#########################   Visualization   #########################
def plt_disp_img(img, name="", save=False):
    plt.axis('off'); plt.margins(0, 0)
    plt.title(name)
    plt.imshow(img, cmap='gray')
    if save:
        name = name if name else 'img'
        path = os.path.join(out_dir(), name + '.png')
        path = get_avail_path(path)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

def cv_disp_img(img, title='', save=False):
    cv2.imshow(title, img); cv2.waitKey(0); cv2.destroyAllWindows()
    if save:
        title = title if title else 'res'
        path = os.path.join(out_dir(), title +'.png')
        path = get_avail_path(path)
        cv2.imwrite(path, img)


    return ('_'+title+'_') if title else ""

def bgr_rgb(img):

    if img.ndim != 3:
        print("error rgb_bgr")
        return img
    return img[:, :, ::-1]
