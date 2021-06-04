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
    dir, base = os.path.split(path)
    name, ext = os.path.splitext(base)
    return dir, name, ext

def get_avail_path(path):
    while os.path.exists(path):
        dir,name,ext = dir_name_ext(path)
        path = os.path.join(dir, name+'0'+ext)
    return path

def get_time_path():
    return datetime.now().strftime("%m_%d_%H_%M")

def fig_path():
    cwd = os.getcwd()
    van_ind = cwd.rfind('VAN_ex')
    base_path = cwd[:van_ind+len('VAN_ex')]
    fig_path = os.path.join(base_path, 'fig')
    return fig_path

def track_path():
    cwd = os.getcwd()
    van_ind = cwd.rfind('VAN_ex')
    base_path = cwd[:van_ind+len('VAN_ex')]
    tracks_path = os.path.join(base_path, 'tracks')
    return tracks_path

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

def rots_matrices_diff(R,Q):
    rvec, _ = cv2.Rodrigues(R.transpose() @ Q)
    radian_diff = np.linalg.norm(rvec)
    deg_diff = radian_diff * 180 / np.pi
    return deg_diff

def get_l0_to_l1_trans_rot(l0, l1):
    r0, t0 = l0[0:3, 0:3], l0[0:3, 3]
    r1, t1 = l1[0:3, 0:3], l1[0:3, 3]
    r0_to_r1 = r1 @ r0.T
    t0_to_t1 = t1 - t0
    return r0_to_r1, t0_to_t1

#########################   Visualization   #########################
def plt_disp_img(img, name, save=False):
    plt.axis('off'); plt.margins(0, 0)
    plt.title(name)
    plt.imshow(img)
    if save:
        path = os.path.join(fig_path(), name + '.png')
        path = get_avail_path(path)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

def cv_disp_img(img, title='', save=False):
    cv2.imshow(title, img); cv2.waitKey(0); cv2.destroyAllWindows()
    if save:
        title = title if title else 'res'
        path = os.path.join(fig_path(), title +'.png')
        path = get_avail_path(path)
        cv2.imwrite(path, img)

#########################   MAYBE   #########################
# features.py
# def kp_desc_scc(img, to_plot=False):
#     NUM_KP = 2000  # number of keypoints to find in an image for ssc
#     fast = cv2.FastFeatureDetector_create()
#     kp = fast.detect(img, mask=None)
#     random.shuffle(kp) # required for ssc
#     kp = ssc.ssc(kp, num_ret_points=NUM_KP, tolerance=0.3, cols=img.shape[1], rows=img.shape[0])
#     if to_plot:
#         img = cv2.drawKeypoints(img, kp, outImage=None, color=(255, 0, 0), flags=0)
#         utils.cv_disp_img(img, title="FAST_keypoints", save=True)
#     brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
#     kp, des = brief.compute(img, kp) # [KeyPoint1, KeyPoint2,.., KeyPoint_n], ndarry (n,32)
#     kp = np.array([keypoint.pt for keypoint in kp]).T # (2,n)
#     return kp, des

# pnp.py
# def pnp_ransac(self, iters=100):
#     best_ext_l1 = None
#     best_ext_l1_inliers_bool = np.zeros(1)
#     largest_inlier_ind = 0
#     best_dists_l1 = None
#
#     for i in range(iters):
#         ext_l1 = self.pnp()
#         if ext_l1 is None: continue
#         inliers_bool, dists_l1 = self.inliers(ext_l1=ext_l1)
#         if sum(inliers_bool) >= sum(best_ext_l1_inliers_bool):
#             largest_inlier_ind = i
#             best_ext_l1 = ext_l1
#             best_ext_l1_inliers_bool = inliers_bool
#             best_dists_l1 = dists_l1
#     print(f"largest inliner found at iter {largest_inlier_ind} with {sum(best_ext_l1_inliers_bool)} inliers")
#     # refine ext_l1 by computing it from all its inlier
#     inlier_pc_l0 = self.pc_l0_r0[:, best_ext_l1_inliers_bool]
#     inlier_kp_l1 = self.kp_l1[:, best_ext_l1_inliers_bool]
#
#     # refine ext_l1 by computing pnp from all its inliers
#     try:
#         tmp_pc_l0, tmp_pxls_l1 = get_pc_pxls_for_cv_pnp(pc_l0_r0=inlier_pc_l0, pxls_l1=inlier_kp_l1, size=inlier_pc_l0.shape[1])
#         retval, rvec, tvec = cv2.solvePnP(objectPoints=tmp_pc_l0, imagePoints=tmp_pxls_l1, cameraMatrix=self.k3, distCoeffs=None)
#         best_ext_l1 = utils.rodrigues_to_mat(rvec, tvec)  # extrinsic (4,4) FROM WORLD (l0) TO CAMERA (l1)
#     except:
#         print("failure in refine best_ext_l1")
#
#     self.best_dists_l1 = best_dists_l1
#     self.best_ext_l1 = best_ext_l1
#     self.best_ext_l1_inliers_bool = best_ext_l1_inliers_bool
#     return best_ext_l1, best_ext_l1_inliers_bool

# triang.py
# def my_triang(pxls1, pxls2, cam_mat1, cam_mat2):
#     """
#     :param pxls1/2: (2,n) ndarray of (x,y) of pixels of matching keypoints in image 1/2
#     :param cam_mat1/2: (3,4) ndarray of projection matrix of camera 1/2
#     :return: (3,n) ndarray
#     """
#     assert pxls1.shape == pxls2.shape
#     assert cam_mat1.shape == cam_mat2.shape == (3,4)
#     num_points = pxls1.shape[1]
#     new_points = np.zeros((4,num_points))
#     p1,p2,p3 = cam_mat1
#     p1_, p2_, p3_ = cam_mat2
#     for i in range(num_points):
#         x,y = pxls1[:,i]
#         x_,y_ = pxls2[:, i]
#         A = np.vstack((p3*x - p1, p3*y - p2, p3_*x_- p1_, p3_*y_ - p2_ ))
#         u,s,vh = np.linalg.svd(A)
#         X = vh[-1,:] # (4,)
#
#         # iterative Least Squares
#         # for j in range(50):
#         #     first_eq = X @ p3
#         #     second_eq = X @ p3_
#         #     B = np.vstack((A[:2]*first_eq, A[2:]*second_eq))
#         #     u, s, vh = np.linalg.svd(B)
#         #     X = vh[-1, :]  # (4,)
#
#         new_points[:,i] = X
#     inhom_points = new_points[:-1,:] / (new_points[-1].reshape(1,-1)) # (3,n)
#     return inhom_points
# def vis_triangulation_compare(img1, cv2_3d, my_3d, pxls1):
#     """ :param cv2_3d: (3,num_of_matches), inhomogeneous
#         :param pxls1: (2,num_of_matches) inhomogeneous """
#     assert cv2_3d.shape[1] == pxls1.shape[1]
#     cv_img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#     my_img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#     num_matches = cv2_3d.shape[1]
#     rand_inds = np.random.randint(0, num_matches, size=5)
#     for ind in rand_inds:
#         x, y = pxls1[:, ind]
#         x = int(x); y = int(y)
#         cv_x_w, cv_y_w, cv_z_w = cv2_3d[:, ind]
#         my_x_w, my_y_w, my_z_w = my_3d[:,ind]
#
#         print(f'({x},{y:}) -> cv:({cv_x_w:.1f},{cv_y_w:.1f},{cv_z_w:.1f}), my:({my_x_w:.1f},{my_y_w:.1f},{my_z_w:.1f})')
#         cv_img1 = cv2.circle(cv_img1, (x, y), 2, (0, 0, 0))
#         cv_img1 = cv2.putText(cv_img1, f'{cv_x_w:.1f},{cv_y_w:.1f},{cv_z_w:.1f}', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 255))
#         my_img1 = cv2.circle(my_img1, (x, y), 2, (0, 0, 0))
#         my_img1 = cv2.putText(my_img1, f'{my_x_w:.1f},{my_y_w:.1f},{my_z_w:.1f}', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 255))
#     cv2.imwrite("1vis_traing_my_com.png", my_img1)
#     cv2.imwrite("1vis_traing_cv2_com.png", cv_img1)
#     # cv_disp_img(img1, title=f"vis_triang_{title}",save=True)