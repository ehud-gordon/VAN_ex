import cv2
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

    utils.vis_triangulation(img1=img1, points_3d=cv2_3d, pts1=pts1, title="cv2")
    utils.vis_triangulation(img1=img1, points_3d=my_3d,  pts1=pts1, title="mine")

if __name__=="__main__":
    img1, img2 = utils.read_images(idx=0)
    kp1, desc1 = kp_desc(img=img1, to_plot=False)
    kp2, desc2 = kp_desc(img=img2, to_plot=False)

    knn_matches = match_desc(desc1=desc1, desc2=desc2)
    stereo_matches, pts1, pts2 = filter_matches(knn_matches, kp1=kp1, kp2=kp2, stereo_filter=True)

    # triangulate
    k, m1, m2 = utils.read_cameras()  # k=(3,3) m1,m2 (3,4)
    p1 = k@m1 # (3,4)
    p2 = k@m2 # (3,4)
    cv2_4d = cv2.triangulatePoints(projMatr1=p1, projMatr2=p2, projPoints1=pts1, projPoints2=pts2) # (4,n)
    cv2_3d = cv2_4d[:-1] / (cv2_4d[-1].reshape(1, -1)) # (3,n)
    my_3d = triangulate(pts1=pts1, pts2=pts2, cam_mat1=p1, cam_mat2=p2)  # (3,n)

    print('end')



