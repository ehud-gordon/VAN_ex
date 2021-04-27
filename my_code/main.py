import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import my_code.utils as utils
import random
import ssc

def kp_desc(img, to_plot=False):
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, mask=None)
    random.shuffle(kp) # required for ssc
    kp = ssc.ssc(kp, num_ret_points=1000, tolerance=0.3, cols=img.shape[1], rows=img.shape[0])
    if to_plot:
        img = cv2.drawKeypoints(img, kp, outImage=None, color=(255, 0, 0), flags=0)
        utils.cv_disp_img(img, title="FAST_keypoints", save=True)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp, des = brief.compute(img, kp)
    return kp, des

def match_desc(desc1,desc2):
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
    matcher = cv2.FlannBasedMatcher(index_params, {})
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)
    return knn_matches

def filter_matches(knn_matches, kp1, kp2, stereo_filter=True):
    pts1, pts2 = [], []
    good_matches = []
    ratio_thresh = 0.7
    for m1, m2 in knn_matches:
        if m1.distance < ratio_thresh * m2.distance:
            y_dist = abs(kp1[m1.queryIdx].pt[1] - kp2[m1.trainIdx].pt[1])
            if stereo_filter and (y_dist > utils.MATCH_Y_DIST_MAX):
                continue
            pts1.append(kp1[m1.queryIdx].pt)
            pts2.append(kp2[m1.trainIdx].pt)
            good_matches.append(m1)
    pts1 = np.array(pts1).T
    pts2 = np.array(pts2).T
    return good_matches, pts1, pts2

def triangulate(pts1, pts2, cam_mat1, cam_mat2):
    """
    :param pts1: 2xn ndarray of (x,y) of pixel of matching keypoints in image 1
    :param pts2: 2xn ndarray of (x,y) of pixel of matching keypoints in image 2
    :param cam_mat1: 3x4 ndarray of projection matrix of camera 1
    :param cam_mat2: 3x4 ndarray of projection matrix of camera 2
    :return:
    """
    assert pts1.shape == pts2.shape
    assert cam_mat1.shape == cam_mat2.shape == (3,4)
    num_points = pts1.shape[1]
    new_points = np.zeros((4,num_points))
    p1,p2,p3 = cam_mat1
    p1_, p2_, p3_ = cam_mat2
    for i in range(num_points):
        x,y = pts1[:,i]
        x_,y_ = pts2[:, i]
        A = np.vstack((p3*x - p1, p3*y - p2, p3_*x_- p1_, p3_*y_ - p2_ ))
        # r1 = p3*x - p1
        # r2 = p3*y - p2
        # r3 = p3_*x_- p1_
        # r4 = p3_*y_ - p2_
        # A = np.array([r1,r2,r3,r4])
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

def filter_cloud_points(points_3d, pts1):
    assert points_3d.shape[1] == pts1.shape[1]
    quantile = np.quantile(points_3d[2, :], q=0.98)
    filter = np.logical_and(points_3d[2, :] > 1, points_3d[2, :] < quantile)
    points_3d = points_3d[:, filter]
    pts1 = pts1[:, filter]

    return points_3d, pts1


def vis_triangulation(img1, points_3d, pts1, title=""):
    """ :param points_3d: (3,num_of_matches), inhomogeneous
        :param pts1: (2,num_of_matches) inhomogeneous """
    assert points_3d.shape[1] == pts1.shape[1]
    points_3d, pts1 = filter_cloud_points(points_3d=points_3d, pts1=pts1)

    # cloud points
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points_3d[0, :], points_3d[2, :], points_3d[1, :]) # this isn't a mistake, plt's z axis is our Y axis
    ax.set_title(title)
    xmin, ymin, zmin = np.min(points_3d, axis=1)
    xmax, ymax, zmax = np.max(points_3d, axis=1)
    ax.set_ylim([0, zmax + 1])  # not a mistake, plt's Y axis is our Z-Axis
    ax.set_xlim([xmin - 1, xmax + 1])
    ax.set_zlim([ymin - 1, ymax + 1]) # not a mistake, plt's z-axis is our Y-axis
    ax.invert_zaxis()  # not a mistake, - plt's z axis is our Y axis
    ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y') # not a mistake
    path = utils.get_avail_path(os.path.join(utils.FIG_PATH, f'cloud_point_{title}.png'))
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    num_matches = points_3d.shape[1]
    rand_inds = np.random.randint(0, num_matches, size=10)
    for ind in rand_inds:
        x_w, y_w, z_w = points_3d[0:3, ind]
        x, y = pts1[:, ind]
        x = int(x); y = int(y)
        print(f'({x},{y:}) -> ({x_w:.1f},{y_w:.1f},{z_w:.1f})')
        img1 = cv2.circle(img1, (x, y), 2, (0, 0, 0))
        img1 = cv2.putText(img1, f'{x_w:.1f},{y_w:.1f},{z_w:.1f}', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                           color=(0, 0, 255), lineType=cv2.LINE_AA)
    print()

    utils.cv_disp_img(img1, title=f"vis_triangulation_{title}",save=True)

def vis_triangulation_compare(img1, cv2_3d, my_3d, pts1):
    """ :param cv2_3d: (3,num_of_matches), inhomogeneous
        :param pts1: (2,num_of_matches) inhomogeneous """
    assert cv2_3d.shape[1] == pts1.shape[1]
    cv_img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    my_img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    num_matches = cv2_3d.shape[1]
    rand_inds = np.random.randint(0, num_matches, size=5)
    for ind in rand_inds:
        x, y = pts1[:, ind]
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

def ex1_main():
    img1, img2 = utils.read_images(idx=0)
    kp1, desc1 = kp_desc(img=img1, to_plot=True)
    kp2, desc2 = kp_desc(img=img2, to_plot=True)
    knn_matches = match_desc(desc1=desc1, desc2=desc2)
    utils.vis_matches(img1, img2, kp1, kp2, knn_matches)

    sig_matches, _, _ = filter_matches(knn_matches, kp1=kp1, kp2=kp2, stereo_filter=False)
    utils.vis_matches(img1, img2, kp1, kp2, sig_matches)

    stereo_matches, pts1, pts2 = filter_matches(knn_matches, kp1=kp1, kp2=kp2, stereo_filter=True)
    utils.vis_matches(img1, img2, kp1, kp2, stereo_matches)

    all_matches = [match[0] for match in knn_matches]
    utils.plot_inliers_outliers(all_matches=all_matches, inliers=stereo_matches, kp1=kp1, kp2=kp2, img1=img1, img2=img2)

    # triangulate
    k, m1, m2 = utils.read_cameras()  # k=(3,3) m1,m2 (3,4)
    p1 = k @ m1
    p2 = k @ m2
    cv2_4d = cv2.triangulatePoints(projMatr1=p1, projMatr2=p2, projPoints1=pts1, projPoints2=pts2)  # (4,n)
    cv2_3d = cv2_4d[:-1] / (cv2_4d[-1].reshape(1, -1))  # (3,n)
    my_3d = triangulate(pts1=pts1, pts2=pts2, cam_mat1=p1, cam_mat2=p2)  # (3,n)

    vis_triangulation(img1=img1, points_3d=cv2_3d, pts1=pts1, title="cv2")
    vis_triangulation(img1=img1, points_3d=my_3d, pts1=pts1, title="mine")

if __name__=="__main__":
    img1, img2 = utils.read_images(idx=0)
    kp1, desc1 = kp_desc(img=img1, to_plot=False)
    kp2, desc2 = kp_desc(img=img2, to_plot=False)

    knn_matches = match_desc(desc1=desc1, desc2=desc2)
    stereo_matches, pts1, pts2 = filter_matches(knn_matches, kp1=kp1, kp2=kp2, stereo_filter=True)

    # triangulate
    k, m1, m2 = utils.read_cameras()  # k=(3,3) m1,m2 (3,4)
    p1 = k@m1
    p2 = k@m2
    cv2_4d = cv2.triangulatePoints(projMatr1=p1, projMatr2=p2, projPoints1=pts1, projPoints2=pts2) # (4,n)
    cv2_3d = cv2_4d[:-1] / (cv2_4d[-1].reshape(1, -1)) # (3,n)
    my_3d = triangulate(pts1=pts1, pts2=pts2, cam_mat1=p1, cam_mat2=p2)  # (3,n)

    vis_triangulation(img1=img1, points_3d=cv2_3d, pts1=pts1, title="cv2")
    vis_triangulation(img1=img1, points_3d=my_3d, pts1=pts1, title="mine")

    print('end')



