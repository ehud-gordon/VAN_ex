import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import scipy.spatial

import os
from itertools import compress

import utils, kitti, triang, my_plot
from utils import CYAN_COLOR, ORANGE_COLOR

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.5g" % x))

LOWE_THRESHOLD = 0.7 # threshold to use in knn matching

class DrawMatchesDouble:
    def __init__(self, img0, img1, kp0, kp1, pc=None, i=0):
        if img0.ndim == 2:
            img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        self.i=i
        self.img0 = img0
        self.img1 = img1
        self.kp0 = kp0
        self.kp1 = kp1
        self.ckdtree0 = scipy.spatial.cKDTree(kp0.T)
        self.ckdtree1 = scipy.spatial.cKDTree(kp1.T)
        self.con_ind = 0
        self.cons = []
        self.curr_cons = []
        self.pc=pc

    def on_hover(self, event):
        if event.inaxes is None:
            return
        if event.inaxes == self.ax0:
            print(event.x)
            print(event.y)
        else:
            pass

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
                [self.ax1.add_artist(con) for con in self.cons]
                self.curr_cons = self.cons
        if event.key == 'n': # draw only next
            self.clear_cons()
            self.curr_cons = [self.cons[self.con_ind]]
            self.ax1.add_artist(self.cons[self.con_ind])
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
        if self.pc is not None: print(f'{self.pc[:,closest_index]}')
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
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        plt.suptitle(f"matches_{matcher_name}_frame={self.i}")
        self.ax0.imshow(self.img0)
        self.ax1.imshow(self.img1)
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
        self.fig.subplots_adjust(left=0.03, bottom=0.19, right=0.99, top=0.94, wspace=0.04, hspace=0.2)
        if save:
            path = os.path.join(utils.out_dir(), f'matches_{matcher_name}' + '.png')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.show()

class Features:
    def __init__(self, det="SIFT", desc="SIFT", matcher="BF", feature_grid=False, save=False, **kwargs):
        self.det = det # detector,
        self.desc = desc # descriptor
        self.matcher_type = matcher
        self.feature_grid = feature_grid
        self.save = save
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
        desc = np.concatenate(new_descs, axis=1)
        print(f"num of kps in all grid:{kp.shape[1]}")
        if self.plot_keypoints:
            kp2 = []
            for (x,y) in kp.T:
                kp2.append(cv2.KeyPoint(x,y,7))
            img = cv2.drawKeypoints(img, kp2, outImage=None, color=(255, 0, 0), flags=0)
            utils.cv_disp_img(img, title=f"{self.det}_{self.desc}_keypoints", save=False)

        return kp, desc

    def kp_desc(self, img, plot_keypoints=False):
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
        if self.det == "STAR" and self.desc == "BRIEF":
            star = cv2.xfeatures2d.StarDetector_create(responseThreshold=10) # default 30,
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            kp = star.detect(img, None)
            kp, desc = brief.compute(img, kp)

        if plot_keypoints:
            img = cv2.drawKeypoints(img, kp, outImage=None, color=(255, 0, 0), flags=0)
            utils.cv_disp_img(img, title=f"{self.det}_{self.desc}_keypoints", save=False)

        kp = np.array([keypoint.pt for keypoint in kp]).T # (2,n)
        # print(f"num of unmatched kps:{kp.shape[1]}")
        return kp, desc.T # (2, n), # (128, n)

    def get_kps_desc_stereo_pair(self, idx):
        img_l0, img_r0 = kitti.read_images(idx=idx)
        return self.get_kps_desc(img_l0, img_r0, is_stereo=True, i=idx)
    
    def get_kps_desc(self,img0, img1, is_stereo, i=0):
        """   :return: kps and descs of (and only of) good matches """
        if self.feature_grid:
            kp0, desc0 = self.kp_desc_grid(img=img0) # (2, n1), ndarray (32, n1)
            kp1, desc1 = self.kp_desc_grid(img=img1) # # (2, n2), ndarray (32, n2)
        else:
            kp0, desc0 = self.kp_desc(img=img0, plot_keypoints=self.plot_keypoints)  # (2, n1), ndarray (32, n1)
            kp1, desc1 = self.kp_desc(img=img1, plot_keypoints=self.plot_keypoints)  # (2, n2), ndarray (32, n2)

        matches = self.matcher.match(queryDescriptors=desc0.T, trainDescriptors=desc1.T)  # list of matches [DMatch1,... DMatch1N]
        matches = filter_matches(matches, kp0=kp0, kp1=kp1, is_stereo=is_stereo)

        kp0_matched, desc0_matched, kp1_matched = filter_with_matches(matches, [kp0, desc0],[kp1])
        if self.plot_matches:
            fig_drawer = DrawMatchesDouble(img0, img1, kp0_matched, kp1_matched, i=i)
            fig_drawer.draw_matches_double(0, save=False, matcher_name=f'{self.matcher_type}_{self.det}_{self.desc}_stereo={is_stereo}')
        return kp0_matched, desc0_matched, kp1_matched

def filter_matches(matches, kp0, kp1, is_stereo):
    good_matches = []
    match_distances = []
    for m in matches:
        y_dist = abs(kp0[1, m.queryIdx] - kp1[1, m.trainIdx])
        if is_stereo and (y_dist > utils.MATCH_Y_DIST_MAX):
            continue
        match_distances.append(m.distance)
        good_matches.append(m)
    match_distances = np.asarray(match_distances)
    # my_plot.plotly_hist(y=match_distances, title=f"match distances, is_stereo={is_stereo}",density=True, plot=False, save=True)
    bool_of_largest = utils.get_perc_largest_indices(match_distances, 0.02)
    matches = list(compress(good_matches, ~bool_of_largest))
    # plt.hist(match_distances, density=True);plt.show()
    return matches

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

def match_two_pairs(i, n):
    featurez = Features()
    k, ext_id, ext_l_to_r = kitti.read_cameras()  # k=(3,4) ext_l0/r0 (4,4)
    kp_li, desc_li, kp_ri = featurez.get_kps_desc_stereo_pair(i)
    kp_li, kp_ri, pc_lr_i_in_li, desc_li = triang.triang_and_rel_filter(kp_li, kp_ri, k, ext_id, ext_l_to_r, desc_li)

    kp_ln, desc_ln, kp_rn = featurez.get_kps_desc_stereo_pair(n)
    kp_ln, kp_rn, pc_lr_n_in_ln, desc_ln = triang.triang_and_rel_filter(kp_ln, kp_rn, k, ext_id, ext_l_to_r, desc_ln)
    # match li-ln
    matches_li_ln = featurez.matcher.match(desc_li.T, desc_ln.T)  # list of matches [DMatch1,... DMatch1N]
    matches_li_ln = filter_matches(matches_li_ln, kp_li, kp_ln, is_stereo=False)

    return kp_li, kp_ri, pc_lr_i_in_li, kp_ln, kp_rn, pc_lr_n_in_ln, matches_li_ln

def filter_with_matches(matches, query_lst, train_lst):
    query_inds = [m.queryIdx for m in matches]
    train_inds = [m.trainIdx for m in matches]
    res = [l[:,query_inds] for l in query_lst]
    res += [l[:,train_inds] for l in train_lst]
    return res


if __name__=="__main__":
    i=570
    n=1330
    import pnp, my_plot, tracks, gtsam_utils, bundle
    from gtsam.symbol_shorthand import X
    import plotly.graph_objects as go
    # pnp match
    k, ext_id, ext_l_to_r = kitti.read_cameras()
    kp_li, kp_ri, pc_lr_i_in_li, kp_ln, kp_rn, pc_lr_n_in_ln, matches_li_ln = match_two_pairs(i, n)
    query_inds = [m.queryIdx for m in matches_li_ln]
    train_inds = [m.trainIdx for m in matches_li_ln]
    kp_li_m, kp_ri_m, pc_i_m, kp_ln_m, kp_rn_m, pc_n_m = filter_with_matches(matches_li_ln, [kp_li, kp_ri,pc_lr_i_in_li], [kp_ln, kp_rn, pc_lr_n_in_ln])
    pnp_ext_ln_to_li, pnp_inliers_bool, _, _ = pnp.pnp_ransac(kp_li_m, kp_ri_m, pc_n_m, k, ext_l_to_r, max_iters=15, frame=f'{n}, {i}')

    # prep small bundelon
    consistent_matches_li_ln = list(compress(matches_li_ln, pnp_inliers_bool))
    tracks_db = tracks.Tracks_DB()
    tracks_db.add_frame(consistent_matches_li_ln, i, n, kp_li, kp_ri, kp_ln, kp_rn, pc_lr_i_in_li, pc_lr_n_in_ln)
    frames_idx = [i, n]
    gt_k = gtsam_utils.get_gt_k(k, ext_l_to_r)
    # do small bundelon
    values, error_before, error_after, bundle_marginals = bundle.do_single_bundle(frames_idx, [ext_id, pnp_ext_ln_to_li], tracks_db, gt_k)
    print(f'bundleon [{i},{n}]: error before: {error_before:.1f}, after: {error_after:.1f}')
    # bundle result
    Pose3_li_to_l0 = values.atPose3( X(i) ); Pose3_ln_to_l0 = values.atPose3( X(n) )
    Pose3_l1330_to_l570 = Pose3_li_to_l0.between(Pose3_ln_to_l0) # bundle result
    ext_1330_to_570 = Pose3_l1330_to_l570.matrix()
    cov_ln_cond_li = gtsam_utils.extract_cov_ln_cond_li_from_marginals(bundle_marginals, i, n)
    cov_li_cond_ln = gtsam_utils.extract_cov_ln_cond_li_from_marginals(bundle_marginals, n, i)


    stg3_pkl = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-10-13-49_0_2760/stage3_40.8_29.9/stage3_ext_lj_to_li_s_cond_covs_2760.pkl'
    stg3_dir, pkl_name, _ = utils.dir_name_ext(stg3_pkl)
    out_dir = os.path.dirname(stg3_dir)
    ext_lj_to_li_s, cov_lj_cond_li_dict, keyframes_idx = gtsam_utils.deserialize_bundle(stg3_pkl, as_ext=True)
    # stage 3 dws
    ext_li_to_l0_s = utils.concat_cj_to_ci_s(ext_lj_to_li_s)
    my_dws = utils.get_dws_from_cam_to_world_s(ext_li_to_l0_s); my_sd = (my_dws, 'mine', 'red')
    new_ext_1330_to_l0_s = ext_li_to_l0_s[57] @ ext_1330_to_570
    new_dw_1330 = utils.get_dws_from_cam_to_world_s([new_ext_1330_to_l0_s]).flatten()
    est_1330 = my_dws[:,133]
    old_570 =my_dws[:,57];old_560 =my_dws[:,56]; old_580 =my_dws[:,58]; old_1320 = my_dws[:,132]; old_1340 = my_dws[:,134]
    # stage 3 covariance
    cov_cj_cond_ci_s = [ cov_lj_cond_li_dict[j][j-1] for j in range(1, len(keyframes_idx)) ]
    cumsum_cov_cj_cond_ci = utils.cumsum_mats(cov_cj_cond_ci_s)
    old_cov_1330_cond_0 = cumsum_cov_cj_cond_ci[132]
    cov_570_cond_0 = cumsum_cov_cj_cond_ci[56]
    new_cov_1330_cond_0 = cov_ln_cond_li + cov_570_cond_0
    # kitti
    kitti_ext_li_to_l0_s = kitti.read_poses_cam_to_world(keyframes_idx)
    kitti_dws = kitti.read_dws(keyframes_idx); kitti_sd = (kitti_dws, 'kitti', 'green')
    dws_names_colors = [my_sd, kitti_sd]
    
    gtsam_utils.plotly_cond_trajectory2(ext_li_to_l0_s,cumsum_cov_cj_cond_ci, "stage3", keyframes_idx, "stage3_elips", plot_dir=os.getcwd(),save=True, plot=False)
    exit()
    # plot
    fig = go.Figure()
    for dw, name, color in dws_names_colors:
        trace = go.Scatter3d(x=dw[0], y=dw[2], z=dw[1], name=name, mode='markers+lines', line=dict(color=color),
                                marker=dict(size=3.5, color=color, opacity=0.5))
        fig.add_trace(trace)
    # add camera axes
    for ext_mat in ext_li_to_l0_s:
        r,t = utils.get_r_t(ext_mat)
        x_axis, y_axis, z_axis = t + (r.T)
        x_line = np.append(t[np.newaxis], x_axis[np.newaxis], axis=0); y_line = np.append(t[np.newaxis], y_axis[np.newaxis], axis=0); z_line = np.append(t[np.newaxis], z_axis[np.newaxis], axis=0)
        fig.add_trace(go.Scatter3d(x=x_line[:,0],y=x_line[:,2],z=x_line[:,1],mode="lines", line=dict(color="red"), showlegend=False))
        fig.add_trace(go.Scatter3d(x=y_line[:,0],y=y_line[:,2],z=y_line[:,1],mode="lines", line=dict(color="green"), showlegend=False))
        fig.add_trace(go.Scatter3d(x=z_line[:,0],y=z_line[:,2],z=z_line[:,1],mode="lines", line=dict(color="blue"), showlegend=False))
    for ext_mat in kitti_ext_li_to_l0_s:
        r,t = utils.get_r_t(ext_mat)
        x_axis, y_axis, z_axis = t + (r.T)
        x_line = np.append(t[np.newaxis], x_axis[np.newaxis], axis=0); y_line = np.append(t[np.newaxis], y_axis[np.newaxis], axis=0); z_line = np.append(t[np.newaxis], z_axis[np.newaxis], axis=0)
        fig.add_trace(go.Scatter3d(x=x_line[:,0],y=x_line[:,2],z=x_line[:,1],mode="lines", line=dict(color="red"), showlegend=False))
        fig.add_trace(go.Scatter3d(x=y_line[:,0],y=y_line[:,2],z=y_line[:,1],mode="lines", line=dict(color="green"), showlegend=False))
        fig.add_trace(go.Scatter3d(x=z_line[:,0],y=z_line[:,2],z=z_line[:,1],mode="lines", line=dict(color="blue"), showlegend=False))
    
    camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.2, y=-2, z=0.5) );fig.update_layout(scene_camera=camera)
    fig.update_layout(showlegend=True)
    fig.update_scenes(zaxis_autorange="reversed");
    # trace = go.Scatter3d(x=[new_dw_1330[0]], y=[new_dw_1330[2]], z=[new_dw_1330[1]], name="1330", mode='markers', marker=dict(color="gold", size=5)) ; fig.add_trace(trace)
    fig.update_traces(textposition='top center')
    fig.update_layout(uniformtext_minsize=3)
    fig.update_layout(scene=dict(
        xaxis_title='X', yaxis_title='Z', zaxis_title='Y', aspectmode='data',
        annotations=[
            # dict(showarrow=True, x=new_dw_1330[0], y=new_dw_1330[2], z=new_dw_1330[1], text="pnp 1330", ay=30, font=dict(size=15)),
            # dict(showarrow=True, x=est_1330[0], y=est_1330[2], z=est_1330[1], text="stage3 1330",ay=-35, font=dict(size=15)),
            # dict(showarrow=True, x=old_570[0], y=old_570[2], z=old_570[1], text="570", ay=-30),
            # dict(showarrow=True, x=old_560[0], y=old_560[2], z=old_560[1], text="560"), dict(showarrow=True, x=old_580[0], y=old_580[2], z=old_580[1], text="580"),
            # dict(showarrow=True, x=old_1320[0], y=old_1320[2], z=old_1320[1], text="1320"), dict(showarrow=True, x=old_1340[0], y=old_1340[2], z=old_1340[1], text="1340"),
            ]))
    # ellipse_old = gtsam_utils.get_ellipse_trace(ext_li_to_l0_s[133], old_cov_1330_cond_0, name="old cov")
    ellipse_old = gtsam_utils.get_ellipse_trace(new_ext_1330_to_l0_s, old_cov_1330_cond_0, name="old cov")
    ellipse_new = gtsam_utils.get_ellipse_trace(new_ext_1330_to_l0_s, new_cov_1330_cond_0, name="new cov")
    # fig.add_traces([ellipse_old, ellipse_new])
    my_plot.plotly_save_fig(fig, f'3D_cams_1330_570', plot_dir="", save=False, plot=True)

    

# if __name__=="__main__":
#     kp1 = np.array([
#         [23.85, 163, 166, 171.3, 175.3, 188.8, 209.3, 229.1, 238, 238.7, 246.2, 274.4, 278.4, 302.5, 313.9, 316.8, 332.2, 351, 385.2, 403.5, 405.6, 466.3, 599.3, 663.6, 675.2, 1111, 1123, 1129],
#         [116.3, 203, 214.5, 159.6, 220.8, 159.6, 199.9, 147.8, 152.3, 180.5, 145.9, 142, 180.5, 196.1, 133.2, 143.7, 153.3, 142.7, 122, 145.7, 154, 201.3, 155.9, 145.6, 144.5, 48.62, 47.85, 47.64]])
#     kp2 = np.array([
#        [10.04, 145.5, 148.3, 159.9, 156.6, 176.6, 189.9, 212.7, 221.4, 228.9, 230.6, 258.5, 269.4, 288.9, 299.4, 301.1, 315.7, 335.1, 371.1, 390.3, 394.2, 455.3, 597, 657.1, 667.2, 1097, 1092, 1097],
#        [116.8, 203, 213.6, 159.9, 220.5, 159.4, 200.6, 148, 152, 180.9, 146.1, 142.1, 180.3, 197, 132.6, 143.6, 153.4, 143.1, 122, 144.2, 154.1, 201.4, 155.5, 145.3, 144, 46.95, 47.43, 46.95]])
#     img1, img2 = kitti.read_images(idx=0)
#     drawer = DrawMatchesDouble(img1, img2, kp1, kp2)
#     drawer.draw_matches_double(size=0)
#
