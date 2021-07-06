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

def sorted_nums_form_dict_keys(d):
    int_keys = [int(k) for k in d.keys()]
    return sorted(int_keys)

def serialize_ext_l0_to_li_s(dir_path, ext_l0_to_li_s, title):
    d = {'ext_l0_to_li_s': ext_l0_to_li_s}
    pkl_path = os.path.join(dir_path, f'ext_l0_to_li_s{lund(title)}.pkl')
    clear_path(pkl_path)
    with open(pkl_path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_path

def deserialize_ext_l0_to_li_s(pkl_path):
    with open(pkl_path, 'rb') as handle:
        d = pickle.load(handle)
    return d['ext_l0_to_li_s']

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

def kp_to_np(kp):
    np_kp = np.array([keypoint.pt for keypoint in kp]).T # (2,n)
    return np_kp

def rotation_matrices_diff(R,Q):
    rvec, _ = cv2.Rodrigues(R.transpose() @ Q)
    radian_diff = np.linalg.norm(rvec)
    deg_diff = radian_diff * 180 / np.pi
    return deg_diff

def comp_ext_mat(ext_mat_1, ext_mat_2):
    rot1, t1 = get_r_t(ext_mat_1)
    rot2, t2 = get_r_t(ext_mat_2)
    rot_diff = rotation_matrices_diff(rot1, rot2)
    trans_diff = np.linalg.norm(t1-t2)
    return rot_diff, trans_diff

def rot_trans_i_to_n(ext_0_to_i, ext_0_to_n): # between
    ext_i_to_0 = inv_extrinsics(ext_0_to_i)
    ext_n_to_0 = inv_extrinsics(ext_0_to_n)
    return rot_trans_A_to_B(ext_i_to_0, ext_n_to_0)

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

def get_r_t(mat):
    r = mat[0:3, 0:3]
    t = mat[0:3,3]
    return r.astype('float64'),t.astype('float64')

def r_t_to_ext(r,t):
    mat = np.hstack((r, t.reshape(3,1)))
    mat = np.vstack((mat, [0,0,0,1]))
    return mat.astype('float64')

def rot_trans_stats(rot_diffs_relative, trans_diffs_relative, endframe):
    rots_total_error = np.sum(rot_diffs_relative)
    rot_avg_error = rots_total_error / endframe
    tx_error, ty_error, tz_error = np.sum(trans_diffs_relative, axis=1)
    trans_total_error = + tx_error + ty_error + tz_error
    trans_avg_error = trans_total_error / endframe
    stats = [f"sum of relative rotation errors over all {endframe} frames = {rots_total_error:.1f} deg",
             f"avg. relative rotation error per frame =  {rots_total_error:.1f}/{endframe} = {rot_avg_error:.2f} deg",
             f"sum of relative translation errors over all {endframe} frames = {trans_total_error:.1f} meters",
             f"avg. relative translation error per frame =  {trans_total_error:.1f}/{endframe} = {trans_avg_error:.2f} meters","\n"
             f"sum of relative translation errors over all {endframe} frames, in x-coordinate {tx_error:.1f} meters",
             f"sum of relative translation errors over all {endframe} frames, in y-coordinate {ty_error:.1f} meters",
             f"sum of relative translation errors over all {endframe} frames, in z-coordinate {tz_error:.1f} meters"]
    return stats, rots_total_error,  trans_total_error

def get_consistent_with_extrinsic(kp_li, kp_ri, pc_lr_i_in_l0, ext_l0_to_li, ext_l_to_r, k):
    """
    The function filter points thus: We have:
    (1) matching pixels in left (kp_li) and right (kp_ri) (we believe with no false matches)
    (2) uncertain locations of their 3D location (in world_left_0 CS),
    (3) Extrinsics matrices: ext_l0_to_li and ext_l_to_r that we belive is correct.
    We project pc_in_l0 to pixels_left_i and pixels_right_i using the ext_mats, to get proj_pixels_left_i and proj_pixels_right_i.
    We then filter, so we keep only those points that're consistent with the ext_mats. That is, their world-point is projected close
    to both their pixels_left_i and pixels_right_i locations.
    :param kp_li/ri: (2,n)  of keypoints in pixels_left_i / pixels_right_i
    :param pc_lr_i_in_l0: (3,n) points in world (left0) CS, matching estimates of kp_li and kp_ri real world locations.
    :param ext_l0_to_li: (4,4) extrinsic matrix from world_left_0 to world_left_i
    :param ext_l_to_r: (4,4) extrinsic matrix from world_left_i to world_right_i
    :param k: (4,4) intrinsics camera matrix
    :return inliers_bool - boolean ndarray (n,), with True in indices of points that're consistent
    """
    assert kp_li.shape[1] == kp_ri.shape[1] == pc_lr_i_in_l0.shape[1]
    if pc_lr_i_in_l0.shape[0] == 3:
        pc_lr_i_in_l0 = np.vstack((pc_lr_i_in_l0, np.ones(pc_lr_i_in_l0.shape[1])))  # (4,n)

    proj_l0_to_li = k @ ext_l0_to_li  # (3,4) # from world_left_0 to pixels_left_i
    ext_l0_to_ri = ext_l_to_r @ ext_l0_to_li  # (4,4) # from world_left_0 to world_right_i
    proj_l0_to_ri = k @ ext_l0_to_ri  # (3,4) # from world_left_0 to pixels_right_i

    # project pc_lr_i_in_l0 to pixels_left_i
    projected_li = proj_l0_to_li @ pc_lr_i_in_l0  # (3,n)
    projected_li = projected_li[0:2] / projected_li[-1]  # (2,n)

    # project pc_lr_i_in_l0 to pixels_right_i
    projected_ri = proj_l0_to_ri @ pc_lr_i_in_l0  # (3,n)
    projected_ri = projected_ri[0:2] / projected_ri[-1]  # (2,n)

    proj_errors_li = np.linalg.norm((kp_li - projected_li), axis=0) # L2 norm
    proj_errors_ri = np.linalg.norm((kp_ri - projected_ri), axis=0)  # L2 norm
    REPROJ_THRESH = 2
    bool_li = proj_errors_ri <= REPROJ_THRESH
    bool_ri = proj_errors_ri <= REPROJ_THRESH
    inliers_bool = bool_li * bool_ri # (n,)
    return inliers_bool, proj_errors_ri, proj_errors_ri

def filt_np(bool_array, *nd_arrays):
    """
    :param bool_array: boolean array of size n
    :param nd_arrays: ndarray of size (?,n)
    :return: the arrays filters
    """
    for arr in nd_arrays:
        assert len(bool_array) == arr.shape[1]
    return [arr[:,bool_array] for arr in nd_arrays]


def filter_with_extrinsics(kp_li, desc_li, kp_ri, pc_in_l0, ext_l0_to_li, ext_l_to_r, k):
    # TODO i probably should use this function
    inliers_bool,_, _ = get_consistent_with_extrinsic(kp_li, kp_ri, pc_in_l0, ext_l0_to_li, ext_l_to_r, k)
    return filt_np(inliers_bool, kp_li, desc_li, kp_ri, pc_in_l0)

def get_rot_trans_diffs(rots_A, rots_B, trans_vecs_A, trans_vecs_B):
    """
    :param rots: list of rotation matrices 
    :param trans_vecs: (3,n) 
    """
    rot_diffs_relative = [rotation_matrices_diff(r, q) for r,q in zip (rots_A, rots_B)]
    rot_diffs_relative = np.array(rot_diffs_relative)
    trans_diffs_relative = np.abs(trans_vecs_A - trans_vecs_B)
    return rot_diffs_relative, trans_diffs_relative

def get_rot_trans_diffs_from_ext_mats2(ext_li_to_l0_A, ext_li_to_l0_B):
    rj_to_ri_s_A, tj_to_ti_s_A  = rot_trans_j_to_i_s(ext_li_to_l0_A)
    rj_to_ri_s_B, tj_to_ti_s_B  = rot_trans_j_to_i_s(ext_li_to_l0_B)
    return get_rot_trans_diffs(rj_to_ri_s_A, rj_to_ri_s_B, tj_to_ti_s_A, tj_to_ti_s_B)

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

if __name__=="__main__":
    import kitti
    kitti_dws = kitti.read_dws() # (3,2761)
    a=3
    pass

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