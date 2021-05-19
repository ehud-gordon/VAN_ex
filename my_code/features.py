import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


import my_code.utils as utils
import ssc
from utils import CYAN_COLOR, ORANGE_COLOR
import kitti

LOWE_THRESHOLD = 0.7 # threshold to use in knn matching
NUM_KP = 2000 # number of keypoints to find in an image for ssc

def kp_desc_scc(img, to_plot=False):
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, mask=None)
    random.shuffle(kp) # required for ssc
    kp = ssc.ssc(kp, num_ret_points=NUM_KP, tolerance=0.3, cols=img.shape[1], rows=img.shape[0])
    if to_plot:
        img = cv2.drawKeypoints(img, kp, outImage=None, color=(255, 0, 0), flags=0)
        utils.cv_disp_img(img, title="FAST_keypoints", save=True)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp, des = brief.compute(img, kp) # [KeyPoint1, KeyPoint2,.., KeyPoint_n], ndarry (n,32)
    kp = np.array([keypoint.pt for keypoint in kp]).T # (2,n)
    return kp, des

def disect_image(img):
    """
    |--------- <x=1226>
    |
    |
    |
    <y=370>
    """
    imgs_origins = []
    y_regions = 2 # along the vertical axis[---]
    x_regions = 4 # along the horizontal axis [ | | | ]
    y_step= img.shape[0]//y_regions # along the vertical axis
    x_step = img.shape[1]//x_regions
    for y in range(y_regions):
        for x in range(x_regions):
            orig = [y_step * y, x_step * x]  # [y,x]
            im = img[y_step*y:y_step*(y+1), x_step*x:x_step*(x+1)]
            imgs_origins.append((im,orig))
    return imgs_origins

def kp_desc_disected(img, to_plot=True):
    imgs_origins = disect_image(img)
    corrected_kps_list = []
    new_descs = []
    for sub_img, origin in imgs_origins:
        orig_kps, desc = kp_desc(sub_img, to_plot=False) # (2,n) in (x,y) format
        if orig_kps.size==0:
            continue
        orig_y, orig_x = origin[0], origin[1] # [y,x]
        corrected_kps = orig_kps + np.array([[orig_x], [orig_y]])
        new_descs.append(desc)
        corrected_kps_list.append(corrected_kps)

    kp = np.concatenate(corrected_kps_list, axis=1)
    desc = np.concatenate(new_descs, axis=0)
    print(f"num of unmatched kps:{kp.shape[1]}")
    if to_plot:
        kp2 = []
        for (x,y) in kp.T:
            kp2.append(cv2.KeyPoint(x,y,7))
        img = cv2.drawKeypoints(img, kp2, outImage=None, color=(255, 0, 0), flags=0)
        utils.cv_disp_img(img, title="FAST_BRIEF_keypoints", save=False)

    return kp, desc


def kp_desc(img, to_plot=True):
    # detector = cv2.FastFeatureDetector_create(threshold=25)
    # descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()


    # detector = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.04, edgeThreshold=10)
    # kp = detector.detect(img, mask=None)  # keypoint.pt = [x,y]

    # AKAZE = cv2.AKAZE_create(); kp, desc = AKAZE.detectAndCompute(img, None)
    # KAZE = cv2.KAZE_create(); kp, desc = KAZE.detectAndCompute(img, None)
    sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10); kp, desc = sift.detectAndCompute(img,None)
    if to_plot:
        img = cv2.drawKeypoints(img, kp, outImage=None, color=(255, 0, 0), flags=0)
        utils.cv_disp_img(img, title="FAST_BRIEF_keypoints", save=False)
    # kp, desc = descriptor.compute(img, kp) # [KeyPoint1, KeyPoint2,.., KeyPoint_n], ndarry (n,32)
    kp = np.array([keypoint.pt for keypoint in kp]).T # (2,n)
    return kp, desc

def match_desc_knn(desc1,desc2):
    """ matches each descriptor in desc1 to two-closest descriptors in desc2
    :param desc1: ndarry (N1, 32) of descriptors of keypoints in img1
    :param desc2: ndarry (N2, 32) of descriptors of keypoints in img2
    :return: [ [DMatch1_1, DMatch1_2], ... , [DMatchN1_1, DMatchN1_2] ]
            list of N1 lists, each list contains 2 DMatch objects, the first one is the closer
    """
    # index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2); matcher = cv2.FlannBasedMatcher(index_params, {})
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matcher = cv2.BFMatcher()
    knn_matches = matcher.knnMatch(desc1, desc2, k=2) # [ [DMatch1_1, DMatch1_2], ... , [DMatchN1_1, DMatchN1_2] ]
    return knn_matches

def filter_knn_matches(knn_matches, kp1, kp2, stereo_filter):
    """ filter matches based on Lowe's threshold and stereo_constraint,
        and returns good matches, and also
    :param knn_matches: (list of N1 lists [ [DMatch1_1, DMatch1_2], ... , [DMatchN1_1, DMatchN1_2] ]
            matching between kp1 (query) and kp2 (train))
    :param kp1/2: (2,n1/2)] in image 1/2, query/train
    :param stereo_filter: boolean, whether to filter based on stereo constraint
    :return: good_matches: [DMatch_1, ..., DMatch_n]
    """
    good_matches = []
    ratio_thresh = LOWE_THRESHOLD
    for m1, m2 in knn_matches:
        if m1.distance < ratio_thresh * m2.distance:
            y_dist = abs(kp1[1, m1.queryIdx] - kp2[1,m1.trainIdx])
            if stereo_filter and (y_dist > utils.MATCH_Y_DIST_MAX):
                continue
            good_matches.append(m1)
    return good_matches

def get_kps_desc_stereo_pair(idx):
    img_l0, img_r0 = kitti.read_images(idx=idx)
    return get_kps_desc(img1=img_l0, img2=img_r0, stereo_filter=True)

def get_kps_desc(img1, img2, stereo_filter):
    """   :return: kps and descs of (and only of) good matches """
    kp1, desc1 = kp_desc(img=img1, to_plot=False)  # (2,n1), ndarray (n1,32)
    kp2, desc2 = kp_desc(img=img2, to_plot=False)  # (2,n2), ndarray (n2,32)
    knn_matches = match_desc_knn(desc1=desc1, desc2=desc2)  # list of N1 lists [ [DMatch1_1, DMatch1_2], ... , [DMatchN1_1, DMatchN1_2] ]
    matches = filter_knn_matches(knn_matches=knn_matches, kp1=kp1, kp2=kp2, stereo_filter=stereo_filter)
    filt_kp1, filt_desc1, filt_kp2  = filter_kp_desc_on_matches(kp1=kp1, kp2=kp2, desc1=desc1, matches=matches)
    # draw_matches_double(img1=img1, img2=img2, kp1=filt_kp1, kp2=filt_kp2,save=False, size=30)
    return filt_kp1, filt_desc1, filt_kp2

def filter_kp_desc_on_matches(kp1, kp2, desc1, matches):
    query_ind = [m.queryIdx for m in matches]
    train_ind = [m.trainIdx for m in matches]
    filt_kp1 = kp1[:, query_ind]
    filt_desc1 = desc1[query_ind]
    filt_kp2 = kp2[:, train_ind]

    return filt_kp1, filt_desc1, filt_kp2
#########################   Visualization Utils   #########################
def draw_matches_double(img1, img2, kp1, kp2,size=0, save=False):
    if isinstance(kp1, list):
        kp1 = get_kp_from_KeyPoints(kp1)
        kp2 = get_kp_from_KeyPoints(kp2)
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.suptitle(f"matches")
    ax1.imshow(img1); ax1.axis('off')
    ax2.imshow(img2); ax2.axis('off')
    inds = range(kp1.shape[1])
    if size:
        inds = np.random.choice(kp1.shape[1], size=size, replace=False)
    for i in inds:
        xy1 = kp1[:,i]
        xy2 = kp2[:,i]
        con = ConnectionPatch(xyA=tuple(xy1), xyB=tuple(xy2), coordsA="data", coordsB="data",
                              axesA=ax1, axesB=ax2)
        ax2.add_artist(con)
        ax1.plot(xy1[0], xy1[1], f'ro', markersize=3)
        ax2.plot(xy2[0], xy2[1], f'ro', markersize=3)
    fig.subplots_adjust(left=0.01, bottom=0.19, right=0.99, top=0.94, wspace=0.01, hspace=0.2)
    if save:
        path = os.path.join(utils.FIG_PATH, 'tmp', f'matches' + '.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_inliers_outliers_keypoints(all_matches, inliers, kp1, kp2, img1, img2):
    outliers = [match for match in all_matches if match not in inliers]

    inliers_kp1 = [kp1[match.queryIdx] for match in inliers]
    outliers_kp1 = [kp1[match.queryIdx] for match in outliers]

    inliers_kp2 = [kp2[match.trainIdx] for match in inliers]
    outliers_kp2 = [kp2[match.trainIdx] for match in outliers]

    img1 = cv2.drawKeypoints(image=img1, keypoints=inliers_kp1,outImage=None, color=ORANGE_COLOR, flags=0)
    img1 = cv2.drawKeypoints(image=img1, keypoints=outliers_kp1, outImage=None, color=CYAN_COLOR, flags=0)
    utils.cv_disp_img(img=img1, title="left image inliers(orange) and outliers(cyan)", save=True)

    img2 = cv2.drawKeypoints(image=img2, keypoints=inliers_kp2, outImage=None, color=ORANGE_COLOR, flags=0)
    img2 = cv2.drawKeypoints(image=img2, keypoints=outliers_kp2, outImage=None, color=CYAN_COLOR, flags=0)
    utils.cv_disp_img(img=img2, title="right image inliers(orange) and outliers(cyan)", save=True)

def get_kp_from_KeyPoints(KeyPoints):
    assert isinstance(KeyPoints, list)
    return np.array([p.pt for p in KeyPoints]).T

def get_Keypoints_from_kp(kp):
    assert isinstance(kp, np.ndarray)
    assert kp.ndim==2; assert kp.shape[0] == 2
    return [cv2.KeyPoint(x,y,7) for (x,y) in kp.T]

def draw_matches_cv(img1, KeyPoints1, img2, KeyPoints2, matches, size=0 ,save=False):
    assert type(KeyPoints1) == type(KeyPoints1) == type([])
    if size:
        inds = np.random.choice(len(matches), size=size, replace=False)
        matches = [matches[i] for i in inds]
    matches_img = cv2.drawMatches(img1, KeyPoints1, img2, KeyPoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    utils.cv_disp_img(img=matches_img, title="matches_20", save=save)

def vis_y_dist_matches_cv(img1, img2, KeyPoints1, KeyPoints2, matches,save=False):
    # get first 20 matches
    if isinstance(matches[0], list):
        matches = [match[0] for match in matches]
    if isinstance(KeyPoints1, np.ndarray):
        KeyPoints1 = get_Keypoints_from_kp(KeyPoints1)
        KeyPoints2 = get_Keypoints_from_kp(KeyPoints2)

    draw_matches_cv(img1=img1, KeyPoints1=KeyPoints1, img2=img2, KeyPoints2=KeyPoints2, matches=matches, size=20,save=save)
    y_dists = []
    for match in matches:
        y_dist = abs(KeyPoints1[match.queryIdx].pt[1] - KeyPoints2[match.trainIdx].pt[1])
        y_dists.append(y_dist)

    # plot histogram
    y_dists = np.array(y_dists)
    plt.hist(y_dists, bins="stone")
    plt.title(f"hist of y-dist of matches,"
              f" {sum(y_dists>2)}/{len(matches)}={sum(y_dists>2)/len(matches):.1%} > 2")
    plt.ylabel("number of matches"); plt.xlabel("match's distance in y-axis")
    hist_path = utils.get_avail_path(os.path.join(utils.FIG_PATH, "matches_hist.png"))
    if save:
        plt.savefig(hist_path)
    plt.show()
    print(f"{sum(y_dists>2)}/{len(matches)}={sum(y_dists>2)/len(matches):.1%} matches with y-dist > 2:")