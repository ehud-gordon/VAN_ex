import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import scipy.spatial

import utils
from utils import CYAN_COLOR, ORANGE_COLOR
import kitti

LOWE_THRESHOLD = 0.7 # threshold to use in knn matching

class Features:
    def __init__(self, args):
        self.det = args.det # detector
        self.desc = args.desc # descriptor
        self.matcher_type = args.matcher
        self.feature_grid = args.feature_grid
        self.save = args.save
        self.plot_keypoints = False
        self.plot_matches = False
        self.plot_grid = False
        self.matcher = self.decide_matcher()

    def decide_matcher(self):
        if self.matcher_type == "BF":
            if self.desc in ["SURF", "SIFT"]:
                return cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
            if self.desc in ["ORB", "BRISK", "BRIEF", "AKAZE"]:
                return cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)

        elif self.matcher_type == "FLANN":
            index_params = None
            search_params = dict(checks=100)  # or pass empty dictionary
            if self.desc in ["SURF", "SIFT"]:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            elif self.desc in ["ORB", "BRISK", "BRIEF", "AKAZE"]:
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                    table_number=6,  # 12
                                    key_size=12,  # 20
                                    multi_probe_level=1)  # 2
            return cv2.FlannBasedMatcher(index_params, search_params)

    def kp_desc_grid(self, img):
        imgs_origins = image_to_grid(img)
        corrected_kps_list = []
        new_descs = []
        for sub_img, origin in imgs_origins:
            orig_kps, desc = self.kp_desc(sub_img, plot_keypoints=self.plot_grid) # (2,n) in (x,y) format
            if orig_kps.size==0:
                continue
            orig_y, orig_x = origin[0], origin[1] # [y,x]
            corrected_kps = orig_kps + np.array([[orig_x], [orig_y]])
            new_descs.append(desc)
            corrected_kps_list.append(corrected_kps)

        kp = np.concatenate(corrected_kps_list, axis=1)
        desc = np.concatenate(new_descs, axis=0)
        print(f"num of unmatched kps in all grid:{kp.shape[1]}")
        if self.plot_keypoints:
            kp2 = []
            for (x,y) in kp.T:
                kp2.append(cv2.KeyPoint(x,y,7))
            img = cv2.drawKeypoints(img, kp2, outImage=None, color=(255, 0, 0), flags=0)
            utils.cv_disp_img(img, title=f"{self.det}_{self.desc}_keypoints", save=False)

        return kp, desc

    def kp_desc(self,img, plot_keypoints=True):
        if self.det == "SURF" and self.desc == "SURF":
            surf_feature2d = cv2.xfeatures2d_SURF.create(hessianThreshold=400) # default 100
            kp, desc = surf_feature2d.detectAndCompute(img, None) # SURF needs L2 NORM

        if self.det == "AKAZE" and self.desc == "AKAZE":
            AKAZE = cv2.AKAZE_create(threshold=0.001) # default 0.001f
            kp, desc = AKAZE.detectAndCompute(img, None)

        if self.det == "SIFT" and self.desc == "SIFT":
            sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10) # nfeatures=0
            # sift = cv2.SIFT_create(contrastThreshold=0.03, edgeThreshold=12)  # nfeatures=0
            kp, desc = sift.detectAndCompute(img,None)

        if plot_keypoints:
            img = cv2.drawKeypoints(img, kp, outImage=None, color=(255, 0, 0), flags=0)
            utils.cv_disp_img(img, title=f"{self.det}_{self.desc}_keypoints", save=False)

        kp = np.array([keypoint.pt for keypoint in kp]).T # (2,n)
        # print(f"num of unmatched kps:{kp.shape[1]}")
        return kp, desc # (2,n), # (n,128)

    def get_kps_desc_stereo_pair(self, idx):
        img_l0, img_r0 = kitti.read_images(idx=idx)
        return self.get_kps_desc(img0=img_l0, img1=img_r0, stereo_filter=True)

    def get_kps_desc(self,img0, img1, stereo_filter):
        """   :return: kps and descs of (and only of) good matches """
        if self.feature_grid:
            kp0, desc0 = self.kp_desc_grid(img=img0) # (2,n1), ndarray (n1,32)
            kp1, desc1 = self.kp_desc_grid(img=img1) # # (2,n2), ndarray (n2,32)
        else:
            kp0, desc0 = self.kp_desc(img=img0, plot_keypoints=self.plot_keypoints)  # (2,n1), ndarray (n1,32)
            kp1, desc1 = self.kp_desc(img=img1, plot_keypoints=self.plot_keypoints)  # (2,n2), ndarray (n2,32)

        # knn_matches = self.matcher.knnMatch(desc1, desc2,k=2)  # [ [DMatch1_1, DMatch1_2], ... , [DMatchN1_1, DMatchN1_2] ]
        # matches = filter_knn_matches(knn_matches=knn_matches, kp1=kp1, kp2=kp2, stereo_filter=stereo_filter)
        matches = self.matcher.match(queryDescriptors=desc0, trainDescriptors=desc1)  # list of matches [DMatch1,... DMatch1N]
        matches = filter_matches(matches, kp0=kp0, kp1=kp1, stereo_filter=stereo_filter)

        filt_kp0, filt_desc0, filt_kp1  = filter_kp_desc_on_matches(kp0=kp0, kp1=kp1, desc0=desc0, matches=matches)
        if self.plot_matches:
            a = DrawMatchesDouble(img0=img0, img1=img1, kp0=filt_kp0, kp1=filt_kp1)
            a.draw_matches_double(size=0, save=False, matcher_name=f'{self.matcher_type}_{self.det}_{self.desc}_stereo={stereo_filter}')
        return filt_kp0, filt_desc0, filt_kp1

def filter_kp_desc_on_matches(kp0, kp1, desc0, matches):
    query_ind = [m.queryIdx for m in matches]
    train_ind = [m.trainIdx for m in matches]
    filt_kp0 = kp0[:, query_ind]
    filt_desc0 = desc0[query_ind]
    filt_kp1 = kp1[:, train_ind]

    return filt_kp0, filt_desc0, filt_kp1

def filter_matches(matches, kp0, kp1, stereo_filter, l1_bad_inds=None):
    good_matches = []
    match_distances = []
    for m in matches:
        y_dist = abs(kp0[1, m.queryIdx] - kp1[1, m.trainIdx])
        if stereo_filter and (y_dist > utils.MATCH_Y_DIST_MAX):
            continue
        match_distances.append(m.distance)
        if m.distance >= 200:
            continue
        if l1_bad_inds and m.queryIdx in l1_bad_inds:
            continue
        good_matches.append(m)
    # my_plot.plotly_hist(y=match_distances, title="match distances",density=True, plot=True, save=False)
    # plt.hist(match_distances, density=True);plt.show()
    return good_matches


def filter_knn_matches(knn_matches, kp0, kp1, stereo_filter):
    """ filter matches based on Lowe's threshold and stereo_constraint,
        and returns good matches
    :param knn_matches: (list of N1 lists [ [DMatch1_1, DMatch1_2], ... , [DMatchN1_1, DMatchN1_2] ]
            matching between kp1 (query) and kp2 (train))
    :param kp0/1: (2,n1/2)] in image 0/1, query/train
    :param stereo_filter: boolean, whether to filter based on stereo constraint
    :return: good_matches: [DMatch_1, ..., DMatch_n]
    """
    good_matches = []
    ratio_thresh = LOWE_THRESHOLD

    # if stereo_filter: # count number
    #     stereo_lowe = 0
    #     stereo_no_lowe = 0
    #     no_stereo_lowe = 0
    #     no_stereo_no_lowe = 0
    #     total = kp1.shape[1]
    #     for m1, m2 in knn_matches:
    #         y_dist = abs(kp1[1, m1.queryIdx] - kp2[1, m1.trainIdx])
    #         if y_dist < utils.MATCH_Y_DIST_MAX:
    #             if m1.distance < ratio_thresh * m2.distance:
    #                 stereo_lowe += 1
    #             else:
    #                 stereo_no_lowe += 1
    #         else:
    #             if m1.distance < ratio_thresh * m2.distance:
    #                 no_stereo_lowe += 1
    #             else:
    #                 no_stereo_no_lowe += 1
    #     assert stereo_lowe + stereo_no_lowe + no_stereo_lowe + no_stereo_no_lowe ==total
    #     print(f'stereo_lowe={stereo_lowe}, stereo_no_lowe={stereo_no_lowe}')
    #     print(f'no_stereo_lowe={no_stereo_lowe}, no_stereo_no_lowe={no_stereo_no_lowe}\n')
    hst = []
    distant_kp0 = []
    distant_kp1 = []
    for i in range(len(knn_matches)):
        if len(knn_matches[i]) != 2:
            continue
        m1, m2 = knn_matches[i]
        if m1.distance < ratio_thresh * m2.distance:
            y_dist = abs(kp0[1, m1.queryIdx] - kp1[1,m1.trainIdx])
            if stereo_filter and (y_dist > utils.MATCH_Y_DIST_MAX):
                continue
            # hst.append(m1.distance)
            if m1.distance >= 200:
                continue
            # if m1.distance >=150:
            #     distant_kp0.append(m1.queryIdx)
            #     distant_kp1.append(m1.trainIdx)
            good_matches.append(m1)
    # plt.hist(hst, density=True);plt.show()
    # distant_kp0 = kp0[:,distant_kp0]; distant_kp1 = kp1[:,distant_kp1]
    return good_matches



def image_to_grid(img):
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


#########################   Visualization Utils   #########################

class DrawMatchesDouble:
    def __init__(self, img0, img1, kp0, kp1):
        if isinstance(kp0, list):
            kp0 = get_kp_from_KeyPoints(kp0)
            kp1 = get_kp_from_KeyPoints(kp1)
        if img0.ndim == 2:
            img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        self.img0 = img0
        self.img1 = img1
        self.kp0 = kp0
        self.kp1 = kp1
        self.ckdtree0 = scipy.spatial.cKDTree(kp0.T)
        self.ckdtree1 = scipy.spatial.cKDTree(kp1.T)
        self.con_ind = 0
        self.cons = []
        self.curr_cons = []

    def clear_cons(self):
        [conp.remove() for conp in self.curr_cons]
        self.curr_cons = []

    def onkeypress(self,event):
        if event.key == 'c': # clear all
            self.clear_cons()
        if event.key == 'd': # draw all
            if len(self.curr_cons) == len(self.cons):
                return
            else:
                self.clear_cons()
                [self.ax2.add_artist(con) for con in self.cons]
                self.curr_cons = self.cons
        if event.key == 'n': # draw only next
            self.clear_cons()
            self.curr_cons = [self.cons[self.con_ind]]
            self.ax2.add_artist(self.cons[self.con_ind])
            self.con_ind  = (self.con_ind + 1) % len(self.cons)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def onmouseclick(self,event):
        if event.inaxes is None:
            return
        if event.inaxes == self.ax1:
            closest_index = self.ckdtree0.query([event.xdata,event.ydata])[1]
        else:
            closest_index = self.ckdtree1.query([event.xdata, event.ydata])[1]
        xy0 = self.kp0[:,closest_index]
        xy1 = self.kp1[:, closest_index]
        self.clear_cons()
        con = ConnectionPatch(xyA=tuple(xy0), xyB=tuple(xy1), coordsA="data", coordsB="data",
                                  axesA=self.ax0, axesB=self.ax1, alpha=0.5, color="blue")
        self.curr_cons = [con]
        self.ax1.add_artist(con)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw_matches_double(self,size=0, save=False, matcher_name=""):
        self.fig, (self.ax0, self.ax1) = plt.subplots(1, 2)
        self.fig.canvas.mpl_connect('button_press_event', self.onmouseclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        plt.suptitle(f"matches_{matcher_name}")
        self.ax0.imshow(self.img0); self.ax0.axis('off')
        self.ax1.imshow(self.img1); self.ax1.axis('off')
        inds = range(self.kp1.shape[1])
        if size:
            inds = np.random.choice(self.kp0.shape[1], size=size, replace=False)
        for i in inds:
            xy0 = self.kp0[:, i]
            xy1 = self.kp1[:, i]
            con = ConnectionPatch(xyA=tuple(xy0), xyB=tuple(xy1), coordsA="data", coordsB="data",
                                  axesA=self.ax0, axesB=self.ax1, alpha=1, color="blue")
            self.cons.append(con)
            self.ax1.add_artist(con)
            self.ax0.plot(xy0[0], xy0[1], f'ro', markersize=3)
            self.ax1.plot(xy1[0], xy1[1], f'ro', markersize=3)
        self.curr_cons = self.cons
        self.fig.subplots_adjust(left=0.01, bottom=0.19, right=0.99, top=0.94, wspace=0.01, hspace=0.2)
        if save:
            path = os.path.join(utils.fig_path(), 'tmp', f'matches_{matcher_name}' + '.png')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.show()

# if __name__=="__main__":
#     kp1 = np.array([
#         [23.85, 163, 166, 171.3, 175.3, 188.8, 209.3, 229.1, 238, 238.7, 246.2, 274.4, 278.4, 302.5, 313.9, 316.8, 332.2, 351, 385.2, 403.5, 405.6, 466.3, 599.3, 663.6, 675.2, 1111, 1123, 1129],
#         [116.3, 203, 214.5, 159.6, 220.8, 159.6, 199.9, 147.8, 152.3, 180.5, 145.9, 142, 180.5, 196.1, 133.2, 143.7, 153.3, 142.7, 122, 145.7, 154, 201.3, 155.9, 145.6, 144.5, 48.62, 47.85, 47.64]])
#     kp2 = np.array([
#        [10.04, 145.5, 148.3, 159.9, 156.6, 176.6, 189.9, 212.7, 221.4, 228.9, 230.6, 258.5, 269.4, 288.9, 299.4, 301.1, 315.7, 335.1, 371.1, 390.3, 394.2, 455.3, 597, 657.1, 667.2, 1097, 1092, 1097],
#        [116.8, 203, 213.6, 159.9, 220.5, 159.4, 200.6, 148, 152, 180.9, 146.1, 142.1, 180.3, 197, 132.6, 143.6, 153.4, 143.1, 122, 144.2, 154.1, 201.4, 155.5, 145.3, 144, 46.95, 47.43, 46.95]])
#     img1, img2 = kitti.read_images(idx=0)
#     a = DrawMatchesDouble(img1, img2, kp1, kp2)
#     a.draw_matches_double(size=0)

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
    hist_path = utils.get_avail_path(os.path.join(utils.fig_path(), "matches_hist.png"))
    if save:
        plt.savefig(hist_path)
    plt.show()
    print(f"{sum(y_dists>2)}/{len(matches)}={sum(y_dists>2)/len(matches):.1%} matches with y-dist > 2:")