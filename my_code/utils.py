import matplotlib.pyplot as plt
import cv2
import numpy as np

import os
from datetime import datetime

FIG_PATH = r'C:\Users\godin\Documents\VAN_ex\fig'

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
    :return: the [R|T]  matrix of global in camera coordinates, samep shape as ext_max
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

#########################   Visualization   #########################
def plt_disp_img(img, name, save=False):
    plt.axis('off'); plt.margins(0, 0)
    plt.title(name)
    plt.imshow(img)
    if save:
        path = os.path.join(FIG_PATH, name + '.png')
        path = get_avail_path(path)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

def cv_disp_img(img, title='', save=False):
    cv2.imshow(title, img); cv2.waitKey(0); cv2.destroyAllWindows()
    if save:
        title = title if title else 'res'
        path = os.path.join(FIG_PATH, title +'.png')
        path = get_avail_path(path)
        cv2.imwrite(path, img)

#########################   MAYBE   #########################
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