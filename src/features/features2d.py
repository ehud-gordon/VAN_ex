"""
Detect keypoints, compute descriptors and  matching.
Supports filtering of matches.
Uses cv2.
"""
import cv2
import numpy as np

from itertools import compress

import utils
from .plot import DrawMatchesDouble
import utils.array

LOWE_THRESHOLD = 0.7 # threshold to use in knn matching
MATCH_Y_DIST_MAX = 2

class Features:
    def __init__(self, detector="SIFT", descriptor="SIFT", matcher="BF", grid=False, save=False, **kwargs):
        self.detector = detector # str
        self.descriptor = descriptor # str
        self.matcher_type = matcher # str
        self.grid = grid # bool
        self.save = save # bool
        self.plot_keypoints = False
        self.plot_matches = False
        self.plot_grid = False
        self.matcher = self.decide_matcher()

    def decide_matcher(self):
        if self.matcher_type == "BF":
            if self.descriptor in ["SURF", "SIFT"]:
                return cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
            if self.descriptor in ["ORB", "BRISK", "BRIEF", "AKAZE"]:
                return cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)

        elif self.matcher_type == "FLANN":
            index_params = None
            search_params = dict(checks=100)
            if self.descriptor in ["SURF", "SIFT"]:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            elif self.descriptor in ["ORB", "BRISK", "BRIEF", "AKAZE"]:
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                    table_number=6,  # 12
                                    key_size=12,  # 20
                                    multi_probe_level=1)  # 2
            return cv2.FlannBasedMatcher(index_params, search_params)

    def keypoints_descriptors_grid(self, img):
        """
        divides an image into cells and finds keypoints and descriptors for each cell.
        returns combined keypoints and descriptors for the entire image.
        """
        grid = utils.image.image_to_grid(img)
        grid_keypoints = []
        grid_descriptors = []
        for cell, origin in grid:
            cell_keypoints, cell_descriptors = self.detectAndCompute(cell, plot_keypoints=self.plot_grid) # (2,n) in (x,y) format
            if cell_keypoints.size==0:
                continue
            orig_y, orig_x = origin[0], origin[1] # [y,x]
            corrected_keypoints = cell_keypoints + np.array([[orig_x], [orig_y]])
            grid_descriptors.append(cell_descriptors)
            grid_keypoints.append(corrected_keypoints)

        grid_keypoints = np.concatenate(grid_keypoints, axis=1)
        grid_descriptors = np.concatenate(grid_descriptors, axis=1)
        print(f"num of combined keypoints in all grid:{grid_keypoints.shape[1]}")
        if self.plot_keypoints:
            cv2_keypoints= [cv2.KeyPoint(x,y,7) for x,y in grid_keypoints.T]
            img = cv2.drawKeypoints(img, cv2_keypoints, outImage=None, color=(255, 0, 0), flags=0)
            utils.image.cv_show_img(img, title=f"{self.detector}_{self.descriptor}_keypoints", save=False)

        return grid_keypoints, grid_descriptors

    def detectAndCompute(self, img, plot_keypoints=False):
        """

        :param bool plot_keypoints: whether to output a plot of the detected keypoints
        :return:
            keypoints: a (2,n) ndarray of format (x,y) of the detected pixels
            descriptors: (m,n) ndarray, where m is descriptor size (128), and n is number of detected keypoints
        """
        if self.detector == "SURF" and self.descriptor == "SURF":
            surf_feature2d = cv2.xfeatures2d_SURF.create(hessianThreshold=400) # default 100
            keypoints, descriptors = surf_feature2d.detectAndCompute(img, None) # SURF needs L2 NORM

        elif self.detector == "AKAZE" and self.descriptor == "AKAZE":
            AKAZE = cv2.AKAZE_create(threshold=0.001) # default 0.001f
            keypoints, descriptors = AKAZE.detectAndCompute(img, None)

        elif self.detector == "SIFT" and self.descriptor == "SIFT":
            sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10) # nfeatures=0
            # sift = cv2.SIFT_create(contrastThreshold=0.03, edgeThreshold=12)  # nfeatures=0
            keypoints, descriptors = sift.detectAndCompute(img,None)
        elif self.detector == "STAR" and self.descriptor == "BRIEF":
            star = cv2.xfeatures2d.StarDetector_create(responseThreshold=10) # default 30,
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
            keypoints = star.detect(img, None)
            keypoints, descriptors = brief.compute(img, keypoints)
        else:
            raise Exception

        if plot_keypoints:
            img = cv2.drawKeypoints(img, keypoints, outImage=None, color=(255, 0, 0), flags=0)
            utils.image.cv_show_img(img, title=f"{self.detector}_{self.descriptor}_keypoints", save=False)

        keypoints = np.array([keypoint.pt for keypoint in keypoints]).T # (2,n)
        return keypoints, descriptors.T # (2, n), # (128, n)

    def detectComputeMatch(self, img1, img2, is_stereo):
        """ finds keypoints and descriptors for two images, matches between them, and
        returns keypoints and desciprotrs of (and only of) good matches

        :param bool is_stereo: are the images rectified and from stereo setup
        :returns: (keypoints1, descriptors1, keypoints2) 
            keypoints1 -  (2,n) ndarray of format (x,y) of the matched keypoints in image 1
            descriptors1 -  (m,n) ndarray, where m is descriptor size (128), and n is number of matched keypoints
            keypoints2 - (2,n) ndarray of format (x,y) of the matched keypoints in image 2
        """
        if self.grid:
            keypoints1, descriptors1 = self.detectAndCompute(img=img1) # (2, n1), ndarray (32, n1)
            keypoints2, descriptors2 = self.detectAndCompute(img=img2) # # (2, n2), ndarray (32, n2)
        else:
            keypoints1, descriptors1 = self.detectAndCompute(img=img1, plot_keypoints=self.plot_keypoints)  # (2, n1), ndarray (32, n1)
            keypoints2, descriptors2 = self.detectAndCompute(img=img2, plot_keypoints=self.plot_keypoints)  # (2, n2), ndarray (32, n2)

        matches = self.Match(keypoints1, descriptors1, keypoints2, descriptors2, is_stereo=is_stereo)

        keypoints1_matched, descriptors1_matched, keypoints2_matched = filter_with_matches(matches, [keypoints1, descriptors1],[keypoints2])
        if self.plot_matches:
            fig_drawer = DrawMatchesDouble(img1, img2, keypoints1_matched, keypoints2_matched)
            fig_drawer.draw_matches_double(0, save=False, title=f'{self.matcher_type}_{self.detector}_{self.descriptor}_is_stereo={is_stereo}')

        return keypoints1_matched, descriptors1_matched, keypoints2_matched


    def Match(self, keypoints1, descriptors1, keypoints2, descriptors2, is_stereo):
        matches = self.matcher.match(descriptors1.T, descriptors2.T)  # list of matches [DMatch1,... DMatch1N]
        return filter_matches(matches, keypoints1, keypoints2, is_stereo)

def filter_matches(matches, keypoints1, keypoints2, is_stereo):
    """ filter matches based on stereo criteria, and also 2 percent of matches with largest match distance"""
    good_matches = []
    match_distances = []
    for m in matches:
        y_dist = abs(keypoints1[1, m.queryIdx] - keypoints2[1, m.trainIdx])
        if is_stereo and (y_dist > MATCH_Y_DIST_MAX):
            continue
        match_distances.append(m.distance)
        good_matches.append(m)
    match_distances = np.asarray(match_distances)
    bool_of_largest = utils.array.get_perc_largest_indices(match_distances, 0.02)
    matches = list(compress(good_matches, ~bool_of_largest))
    return matches

def filter_with_matches(matches, query_lst, train_lst):
    """ Filer np arrays using a list of matches [DMatch1, ...DMatchN] """
    query_inds = [m.queryIdx for m in matches]
    train_inds = [m.trainIdx for m in matches]
    res = [l[:,query_inds] for l in query_lst]
    res += [l[:,train_inds] for l in train_lst]
    return res


# if __name__=="__main__":
#     i=570
#     n=1330
#     from pgm.tracks import tracks
#     from utils import plot
#     from pgm import bundle, utils
#     from calib3d import pnp
#     from gtsam.symbol_shorthand import X
#     import plotly.graph_objects as go
#     # pnp match
#     k, ext_id, ext_l_to_r = kitti.read_cameras()
#     kp_li, kp_ri, pc_lr_i_in_li, kp_ln, kp_rn, pc_lr_n_in_ln, matches_li_ln = match_two_pairs(i, n)
#     query_inds = [m.queryIdx for m in matches_li_ln]
#     train_inds = [m.trainIdx for m in matches_li_ln]
#     kp_li_m, kp_ri_m, pc_i_m, kp_ln_m, kp_rn_m, pc_n_m = filter_with_matches(matches_li_ln, [kp_li, kp_ri,pc_lr_i_in_li], [kp_ln, kp_rn, pc_lr_n_in_ln])
#     pnp_ext_ln_to_li, pnp_inliers_bool, _, _ = pnp.pnp_stereo_ransac(kp_li_m, kp_ri_m, pc_n_m, k, ext_l_to_r, max_iters=15, frame=f'{n}, {i}')

#     # prep small bundelon
#     consistent_matches_li_ln = list(compress(matches_li_ln, pnp_inliers_bool))
#     tracks_db = tracks.Tracks_DB()
#     tracks_db.add_frame(consistent_matches_li_ln, i, n, kp_li, kp_ri, kp_ln, kp_rn, pc_lr_i_in_li, pc_lr_n_in_ln)
#     frames_idx = [i, n]
#     gt_k = utils.get_gt_k(k, ext_l_to_r)
#     # do small bundelon
#     values, error_before, error_after, bundle_marginals = bundle.do_single_bundle(frames_idx, [ext_id, pnp_ext_ln_to_li], tracks_db, gt_k)
#     print(f'bundleon [{i},{n}]: error before: {error_before:.1f}, after: {error_after:.1f}')
#     # bundle result
#     Pose3_li_to_l0 = values.atPose3( X(i) ); Pose3_ln_to_l0 = values.atPose3( X(n) )
#     Pose3_l1330_to_l570 = Pose3_li_to_l0.between(Pose3_ln_to_l0) # bundle result
#     ext_1330_to_570 = Pose3_l1330_to_l570.matrix()
#     cov_ln_cond_li = utils.extract_cov_ln_cond_li_from_marginals(bundle_marginals, i, n)
#     cov_li_cond_ln = utils.extract_cov_ln_cond_li_from_marginals(bundle_marginals, n, i)


#     stg3_pkl = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-10-13-49_0_2760/stage3_40.8_29.9/stage3_ext_lj_to_li_s_cond_covs_2760.pkl'
#     stg3_dir, pkl_name, _ = sys_utils.dir_name_ext(stg3_pkl)
#     out_dir = os.path.dirname(stg3_dir)
#     ext_lj_to_li_s, cov_lj_cond_li_dict, keyframes_idx = utils.deserialize_bundle(stg3_pkl, as_ext=True)
#     # stage 3 dws
#     ext_li_to_l0_s = utils.geometry.concat_cj_to_ci_s(ext_lj_to_li_s)
#     my_dws = utils.geometry.get_dws_from_cam_to_world_s(ext_li_to_l0_s); my_sd = (my_dws, 'mine', 'red')
#     new_ext_1330_to_l0_s = ext_li_to_l0_s[57] @ ext_1330_to_570
#     new_dw_1330 = utils.geometry.get_dws_from_cam_to_world_s([new_ext_1330_to_l0_s]).flatten()
#     est_1330 = my_dws[:,133]
#     old_570 =my_dws[:,57];old_560 =my_dws[:,56]; old_580 =my_dws[:,58]; old_1320 = my_dws[:,132]; old_1340 = my_dws[:,134]
#     # stage 3 covariance
#     cov_cj_cond_ci_s = [ cov_lj_cond_li_dict[j][j-1] for j in range(1, len(keyframes_idx)) ]
#     cumsum_cov_cj_cond_ci = utils.array.cumsum_mats(cov_cj_cond_ci_s)
#     old_cov_1330_cond_0 = cumsum_cov_cj_cond_ci[132]
#     cov_570_cond_0 = cumsum_cov_cj_cond_ci[56]
#     new_cov_1330_cond_0 = cov_ln_cond_li + cov_570_cond_0
#     # kitti
#     kitti_ext_li_to_l0_s = kitti.read_poses_cam_to_world(keyframes_idx)
#     kitti_dws = kitti.read_dws(keyframes_idx); kitti_sd = (kitti_dws, 'kitti', 'green')
#     dws_names_colors = [my_sd, kitti_sd]
    
#     utils.plotly_cond_trajectory2(ext_li_to_l0_s, cumsum_cov_cj_cond_ci, "stage3", keyframes_idx, "stage3_elips", plot_dir=os.getcwd(), save=True, plot=False)
#     exit()
    # plot
    fig = go.Figure()
    for dw, name, color in dws_names_colors:
        trace = go.Scatter3d(x=dw[0], y=dw[2], z=dw[1], name=name, mode='markers+lines', line=dict(color=color),
                                marker=dict(size=3.5, color=color, opacity=0.5))
        fig.add_trace(trace)
    # add camera axes
    for ext_mat in ext_li_to_l0_s:
        r,t = utils.utils_geometry.get_rot_trans(ext_mat)
        x_axis, y_axis, z_axis = t + (r.T)
        x_line = np.append(t[np.newaxis], x_axis[np.newaxis], axis=0); y_line = np.append(t[np.newaxis], y_axis[np.newaxis], axis=0); z_line = np.append(t[np.newaxis], z_axis[np.newaxis], axis=0)
        fig.add_trace(go.Scatter3d(x=x_line[:,0],y=x_line[:,2],z=x_line[:,1],mode="lines", line=dict(color="red"), showlegend=False))
        fig.add_trace(go.Scatter3d(x=y_line[:,0],y=y_line[:,2],z=y_line[:,1],mode="lines", line=dict(color="green"), showlegend=False))
        fig.add_trace(go.Scatter3d(x=z_line[:,0],y=z_line[:,2],z=z_line[:,1],mode="lines", line=dict(color="blue"), showlegend=False))
    for ext_mat in kitti_ext_li_to_l0_s:
        r,t = utils.utils_geometry.get_rot_trans(ext_mat)
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
    ellipse_old = utils.get_ellipse_trace(new_ext_1330_to_l0_s, old_cov_1330_cond_0, name="old cov")
    ellipse_new = utils.get_ellipse_trace(new_ext_1330_to_l0_s, new_cov_1330_cond_0, name="new cov")
    # fig.add_traces([ellipse_old, ellipse_new])
    plot.plotly_save_fig(fig, f'3D_cams_1330_570', plot_dir="", save=False, plot=True)