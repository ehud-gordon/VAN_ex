import re
import shutil

import matplotlib.pyplot as plt
import cv2
import numpy as np

import os
from datetime import datetime

FIG_PATH = os.path.join('C:', 'Users', 'godin', 'Documents', 'VAN_ex', 'fig')
FIG_PATH_WSL = os.path.join(os.sep, 'mnt','c','Users','godin','Documents','VAN_ex', 'fig')

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

def make_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)

def make_avail_path(path):
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
    fig_path = os.path.join(base_path, 'out')
    return fig_path

def path_to_linux(path):
    parts = re.split(r'\\', path)
    if len(parts) == 1: return path
    right_parts = ['/mnt']
    for p in parts:
        if p=='C:':
            p = 'c'
        right_parts.append(p)
    return r'/'.join(right_parts)
#########################   Geometry   #########################
def rodrigues_to_mat(rvec,tvec):
    rot, _ = cv2.Rodrigues(src=rvec)
    extrinsic = np.hstack((rot, tvec))
    extrinsic = np.vstack((extrinsic, np.array([0,0,0,1])))
    return extrinsic # (4,4)

def get_dw_from_extrinsics(ext_mat):
    """
    :param ext_mat: [R|t] extrinsics matrix of some camera
    :return: (3,1) vector in global (world) coordinates of camera origin
    """
    assert ext_mat.shape in [(3,4), (4,4)]
    r,t = ext_mat[:3,:3], ext_mat[0:3,-1]
    return r.T @ -t

def inv_extrinsics(ext_mat):
    """
    :param ext_mat: [R|t] extrinsics matrix of some camera in global coordinates
    :return: the [R|T]  matrix of global in camera coordinates, same shape as ext_max
    """
    assert ext_mat.shape in [(3, 4), (4, 4)]
    r,t = ext_mat[:3,:3], ext_mat[0:3,-1]
    inv = np.hstack((r.T, (r.T@-t).reshape(3,1)))
    if ext_mat.shape == (4,4):
        inv = np.vstack((inv, np.array([0,0,0,1])))
    return inv

def kp_to_np(kp):
    np_kp = np.array([keypoint.pt for keypoint in kp]).T # (2,n)
    return np_kp

def rotation_matrices_diff(R,Q):
    rvec, _ = cv2.Rodrigues(R.transpose() @ Q)
    radian_diff = np.linalg.norm(rvec)
    deg_diff = radian_diff * 180 / np.pi
    return deg_diff

def r0_to_r1_t0_to_t1(l0, l1):
    r0, t0 = l0[0:3, 0:3], l0[0:3, 3]
    r1, t1 = l1[0:3, 0:3], l1[0:3, 3]
    r0_to_r1 = r1 @ r0.T
    t0_to_t1 = t1 - t0
    return r0_to_r1, t0_to_t1

def r0_to_r1_s_t0_to_t1_s(ext_mats):
    r0_to_r1_s, t0_to_t1_s = [], []
    for i in range(1, len(ext_mats)):
        r0_to_r1, t0_to_t1 = r0_to_r1_t0_to_t1(l0=ext_mats[i-1], l1=ext_mats[i])
        r0_to_r1_s.append(r0_to_r1)
        t0_to_t1_s.append(t0_to_t1)
    t0_to_t1_s = np.array(t0_to_t1_s).T
    return r0_to_r1_s, t0_to_t1_s

def r0_to_r1_s_t0_to_t1_s_2(rot_mats, trans_vecs):
    assert len(rot_mats) == trans_vecs.shape[1]
    r0_to_r1_s, t0_to_t1_s = [], []
    for i in range(1, len(rot_mats)):
        r0_to_r1 = rot_mats[i] @ rot_mats[i-1].T
        r0_to_r1_s.append(r0_to_r1)
        t0_to_t1 = trans_vecs[i] - trans_vecs[i-1]
        t0_to_t1_s.append(t0_to_t1)
    t0_to_t1_s = np.array(t0_to_t1_s).T
    return r0_to_r1_s, t0_to_t1_s

def rot_trans_stats(rot_diffs_relative, trans_diffs_relative, endframe):
    rots_total_error = np.sum(rot_diffs_relative)
    rot_avg_error = rots_total_error / endframe
    tx_error, ty_error, tz_error = np.sum(trans_diffs_relative, axis=1)
    trans_total_error = + tx_error + ty_error + tz_error
    trans_avg_error = trans_total_error / endframe
    stats = [f"sum of relative rotation errors over all {endframe} frames = {rots_total_error:.1f} deg",
             f"relative rotation error per frame =  {rots_total_error:.1f}/{endframe} = {rot_avg_error:.2f} deg",
             f"avg. relative translation error = {trans_avg_error:.2f} meters",
             f"total translation error = {trans_total_error:.1f} meters",
             f"tx total error:{tx_error:.1f} meters",
             f"ty total error:{ty_error:.1f} meters",
             f"tz total error:{tz_error:.1f} meters"]
    return stats, rots_total_error,  trans_total_error

def get_consistent_with_extrinsic(kp_li, kp_ri, pc_in_l0, ext_l0_li, ext_li_ri, k):
    """
    filter points thus: we take points in world_left_0, that we know their place in pixels_left_i and pixels_right_i,
    and take only those points who're consistent with ext_l0_li. That is, their world-point is projected close
    to both pixels_left_i and pixels_right_i locations.
    :param kp_li/ri: (2,n)
    :param pc_in_l0: (3,n) points in world_left_0 coordinate system, that we know their kp_li and kp_ri locations.
    :param ext_l0_li: (4,4) extrinsic matrix from world_left_0 to world_left_i
    :param ext_li_ri: (4,4) extrinsic matrix from world_left_i to world_right_i
    :param k: (4,4) intrinsics camera matrix
    :return inliers_bool - boolean array of
    """
    assert kp_li.shape[1] == kp_ri.shape[1] == pc_in_l0.shape[1]
    if pc_in_l0.shape[0] == 3:
        pc_in_l0 = np.vstack((pc_in_l0, np.ones(pc_in_l0.shape[1])))  # (4,n)

    proj_wl0_pli = k @ ext_l0_li  # (3,4) # from world_left_0 to pixels_left_i
    ext_l0_ri = ext_li_ri @ ext_l0_li  # (4,4) # from world_left_0 to world_right_i
    proj_wl0_pri = k @ ext_l0_ri  # (3,4) # from world_left_0 to pixels_right_i

    # project pc_in_l0 to pixels_left_i
    projected_li = proj_wl0_pli @ pc_in_l0  # (3,n)
    projected_li = projected_li[0:2] / projected_li[-1]  # (2,n)

    # project pc_in_l0 to pixels_right_i
    projected_ri = proj_wl0_pri @ pc_in_l0  # (3,n)
    projected_ri = projected_ri[0:2] / projected_ri[-1]  # (2,n)

    projections_errors_to_li = np.sqrt(np.sum((kp_li - projected_li) ** 2, axis=0))
    projections_errors_to_ri = np.sqrt(np.sum((kp_ri - projected_ri) ** 2, axis=0))
    REPROJ_THRESH = 2
    bool_li = projections_errors_to_li <= REPROJ_THRESH
    bool_ri = projections_errors_to_ri <= REPROJ_THRESH
    inliers_bool = bool_li * bool_ri # (n,)
    return inliers_bool, projections_errors_to_li

def filter_with_extrinsic(kp_l1, desc_l1, kp_r1, pc_in_l0, ext_l0_l1, ext_li_ri, k):
    inliers_bool,_ = get_consistent_with_extrinsic(kp_li=kp_l1, kp_ri=kp_r1, pc_in_l0=pc_in_l0, ext_l0_li=ext_l0_l1, ext_li_ri=ext_li_ri, k=k)
    kp_l1 = kp_l1[:,inliers_bool]
    desc_l1 = desc_l1[inliers_bool,:]
    kp_r1 = kp_r1[:, inliers_bool]
    pc_in_l0 = pc_in_l0[:, inliers_bool]
    return kp_l1, desc_l1, kp_r1, pc_in_l0

#########################   Visualization   #########################
def plt_disp_img(img, name, save=False):
    plt.axis('off'); plt.margins(0, 0)
    plt.title(name)
    plt.imshow(img)
    if save:
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