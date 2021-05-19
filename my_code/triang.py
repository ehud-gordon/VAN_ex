import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import my_code.utils as utils

def get_point_cloud_filter(pc):
    quantile = np.quantile(pc[2, :], q=0.98)
    filtered = np.logical_and(pc[2, :] > 1, pc[2, :] < quantile)
    return filtered

def my_triang(pxls1, pxls2, cam_mat1, cam_mat2):
    """
    :param pxls1/2: (2,n) ndarray of (x,y) of pixels of matching keypoints in image 1/2
    :param cam_mat1/2: (3,4) ndarray of projection matrix of camera 1/2
    :return: (3,n) ndarray
    """
    assert pxls1.shape == pxls2.shape
    assert cam_mat1.shape == cam_mat2.shape == (3,4)
    num_points = pxls1.shape[1]
    new_points = np.zeros((4,num_points))
    p1,p2,p3 = cam_mat1
    p1_, p2_, p3_ = cam_mat2
    for i in range(num_points):
        x,y = pxls1[:,i]
        x_,y_ = pxls2[:, i]
        A = np.vstack((p3*x - p1, p3*y - p2, p3_*x_- p1_, p3_*y_ - p2_ ))
        u,s,vh = np.linalg.svd(A)
        X = vh[-1,:] # (4,)

        # iterative Least Squares
        # for j in range(50):
        #     first_eq = X @ p3
        #     second_eq = X @ p3_
        #     B = np.vstack((A[:2]*first_eq, A[2:]*second_eq))
        #     u, s, vh = np.linalg.svd(B)
        #     X = vh[-1, :]  # (4,)

        new_points[:,i] = X
    inhom_points = new_points[:-1,:] / (new_points[-1].reshape(1,-1)) # (3,n)
    return inhom_points

def triang(kp1, kp2, k, m1, m2):
    """
    get 3D (World) coordinates via triangulation of matched pixels from two images
    :param kp1/2: pixels of matched points in image1/2, (2,n) array
    :param k: instrinsics  matrix shared by camera 1/2, (3,4)
    :param m1/2: extrinsics matrices of camera 1/2 (4,4)
    :return: (3,n) ndarray of world coordinates relative to camera1 (left camera)
    """
    assert kp1.shape == kp2.shape
    p1 = k @ m1  # (3,4) # projection matrix
    p2 = k @ m2  # (3,4)

    pc_4d = cv2.triangulatePoints(projMatr1=p1, projMatr2=p2, projPoints1=kp1, projPoints2=kp2)  # (4,n)
    pc_3d = pc_4d[:-1] / (pc_4d[-1].reshape(1, -1))  # (3,n)
    return pc_3d

def triang_from_keypoints(kp1, kp2, k, m1, m2):
    """ :param kp1/2: [KeyPoint1, .... , KeypointN] """
    assert len(kp1) == len(kp2)
    pxls1 = np.array([kp.pt for kp in kp1]).T  # (2,n_matches)
    pxls2 = np.array([kp.pt for kp in kp2]).T  # (2,n_matches)
    pc_3d = triang(kp1=pxls1, kp2=pxls2, k=k, m1=m1, m2=m2)  # (3,n_matches)
    return pc_3d

#########################   Visualization utils  #########################
def vis_pc(pc, title="", color="blue", save=False):
    """ :param pc: (3,num_of_matches), inhomogeneous"""
    assert pc.shape[0] in [3,4]
    if pc.shape[0] == 4:
        pc = pc[:3] / pc[-1,:]
    pc_filter = get_point_cloud_filter(pc=pc)
    pc = pc[:, pc_filter]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pc[0, :], pc[2, :], pc[1, :], color="blue") # this isn't a mistake, plt's z axis is our Y axis
    ax.set_title(title)
    xmin, ymin, zmin = np.min(pc, axis=1)
    xmax, ymax, zmax = np.max(pc, axis=1)
    ax.set_ylim([0, zmax + 1])  # not a mistake, plt's Y axis is our Z-Axis
    ax.set_xlim([xmin - 1, xmax + 1])
    ax.set_zlim([ymin - 1, ymax + 1]) # not a mistake, plt's z-axis is our Y-axis
    ax.invert_zaxis()  # not a mistake, - plt's z axis is our Y axis
    ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y') # not a mistake
    plt.legend()
    if save:
        path = utils.get_avail_path(os.path.join(utils.FIG_PATH, f'{title}_pc.png'))
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

def vis_triang(img, pc, pxls, title="", save=False):
    """ :param pc: (3,num_of_matches), inhomogeneous
        :param pxls: (2,num_of_matches) inhomogeneous """
    assert pc.shape[1] == pxls.shape[1]
    vis_pc(pc=pc,title=title, save=save)
    pc_filter = get_point_cloud_filter(pc=pc)
    pc = pc[:,pc_filter]
    pxls = pxls[:,pc_filter]

    # plot pxls with txt indicating their 3d location
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    num_matches = pc.shape[1]
    rand_inds = np.random.randint(0, num_matches, size=10)
    for ind in rand_inds:
        x_w, y_w, z_w = pc[0:3, ind]
        x, y = pxls[:, ind]
        x = int(x); y = int(y)
        print(f'({x},{y:}) -> ({x_w:.1f},{y_w:.1f},{z_w:.1f})')
        img = cv2.circle(img, center=(x, y), radius=2, color=(0, 0, 0),thickness=1)
        img = cv2.putText(img, f'{x_w:.1f},{y_w:.1f},{z_w:.1f}', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                           color=(0, 0, 255), lineType=cv2.LINE_AA)
    print()
    utils.cv_disp_img(img, title=f"{title}_vis_tr",save=save)

def vis_triangulation_compare(img1, cv2_3d, my_3d, pxls1):
    """ :param cv2_3d: (3,num_of_matches), inhomogeneous
        :param pxls1: (2,num_of_matches) inhomogeneous """
    assert cv2_3d.shape[1] == pxls1.shape[1]
    cv_img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    my_img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    num_matches = cv2_3d.shape[1]
    rand_inds = np.random.randint(0, num_matches, size=5)
    for ind in rand_inds:
        x, y = pxls1[:, ind]
        x = int(x); y = int(y)
        cv_x_w, cv_y_w, cv_z_w = cv2_3d[:, ind]
        my_x_w, my_y_w, my_z_w = my_3d[:,ind]

        print(f'({x},{y:}) -> cv:({cv_x_w:.1f},{cv_y_w:.1f},{cv_z_w:.1f}), my:({my_x_w:.1f},{my_y_w:.1f},{my_z_w:.1f})')
        cv_img1 = cv2.circle(cv_img1, (x, y), 2, (0, 0, 0))
        cv_img1 = cv2.putText(cv_img1, f'{cv_x_w:.1f},{cv_y_w:.1f},{cv_z_w:.1f}', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 255))
        my_img1 = cv2.circle(my_img1, (x, y), 2, (0, 0, 0))
        my_img1 = cv2.putText(my_img1, f'{my_x_w:.1f},{my_y_w:.1f},{my_z_w:.1f}', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 255))
    cv2.imwrite("1vis_traing_my_com.png", my_img1)
    cv2.imwrite("1vis_traing_cv2_com.png", cv_img1)
    # cv_disp_img(img1, title=f"vis_triang_{title}",save=True)

def kp_desc(img, to_plot=True):
    detector = cv2.SIFT_create(contrastThreshold=0.05, edgeThreshold=9)
    kp = detector.detect(img, mask=None) # keypoint.pt = [x,y]
    if to_plot:
        img = cv2.drawKeypoints(img, kp, outImage=None, color=(255, 0, 0), flags=0)
        utils.cv_disp_img(img, title="FAST_BRIEF_keypoints", save=False)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp, desc = brief.compute(img, kp) # [KeyPoint1, KeyPoint2,.., KeyPoint_n], ndarry (n,32)
    kp = np.array([keypoint.pt for keypoint in kp]).T # (2,n)
    return kp, desc

if __name__=="__main__":
    import kitti
    import features
    img_l0, img_r0 = kitti.read_images(0)
    kp_l0, desc_l0, kp_r0 = features.get_kps_desc_stereo_pair(idx=0)
    k, l0_ext, r0_ext = kitti.read_cameras()
    pc_3d = triang(kp1=kp_l0, kp2=kp_r0, k=k, m1=l0_ext, m2=r0_ext)
    vis_triang(img=img_l0,pc=pc_3d, pxls=kp_l0,title="a",save=False)
