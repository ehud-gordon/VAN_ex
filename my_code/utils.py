import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
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