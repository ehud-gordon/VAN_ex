import numpy as np
import cv2

### Filtering keypoints ####
init_ext_l0_l1 = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

init_ext_l0_r0 = np.array([[1, 0, 0, -0.5372],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

k = np.array([[707.1, 0, 601.9, 0],
              [0, 707.1, 183.1, 0],
              [0, 0, 1, 0]])


def get_keypoints_for_stereo_pair(frame_idx):
    # this returns keypoints after matching left and right
    keypoints_left = np.array([])
    keypoints_right = np.array([])
    return keypoints_left, keypoints_right

keypoints_left, keypoints_right = get_keypoints_for_stereo_pair(frame_idx)  # each is ndrray (2,n)

### use simple triangulation to filter bad points ###

proj_left = k @ init_ext_l0_l1
proj_right = k @ init_ext_l0_r0
temp_pc_4d = cv2.triangulatePoints(proj_left, proj_right, keypoints_left, keypoints_right)  # ndarray (4,n)
temp_pc_3d = temp_pc_4d[0:3] / temp_pc_4d[-1]  # ndarray (3,n)


def get_relative_point_cloud_filter(pc):
    """ :param pc: ndarray (3,n) """
    x_abs = np.abs(pc[0])
    y_abs = np.abs(pc[1])
    x_crit = (x_abs <= 50)
    y_crit = (y_abs <= 50)
    z_crit1 = pc[2] < 200
    z_crit2 = pc[2] > 1
    z_crit = z_crit1 * z_crit2
    filter_bool = (x_crit * y_crit) * z_crit
    return filter_bool  # boolean array (n,)

filter_bool = get_relative_point_cloud_filter(temp_pc_3d)
keypoints_left_filtered = keypoints_left[:, filter_bool]
keypoints_right_filtered = keypoints_right[:, filter_bool]