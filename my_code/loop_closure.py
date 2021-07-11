import numpy as np
from numpy import pi
import gtsam
from gtsam import Pose3
from gtsam.symbol_shorthand import X
from plotly import graph_objects
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import os, time
import copy
from itertools import compress
from collections import defaultdict
import warnings; warnings.filterwarnings("ignore")
from operator import itemgetter

import kitti, tracks, utils, my_plot, results, features, bundle, pnp, pose_graph
import shortest_path
import gtsam_utils as g_utils

MAHAL_THRESH = 0.05 # when to check for inliers
INLIERS_PERC_LC_THRESH = 0.6

class LoopClosure:
    def __init__(self, pkl_path):
        # Folders
        self.stage3_pkl_path = utils.path_to_current_os(pkl_path)
        self.stage3_dir, _, _ = utils.dir_name_ext(self.stage3_pkl_path)
        self.out_path = os.path.dirname(self.stage3_dir)
        self.stage5_dir = os.path.join(self.out_path, 'stage5')
        utils.clear_and_make_dir(self.stage5_dir)
        self.num_of_lc = 0
        self.lc_msg = 'lc_0'
        # init kitti
        # Extract stage3 results
        s3_Pose3_lj_to_li_list, self.s3_cov_ln_cond_li_dict, self.keyframes_idx = g_utils.deserialize_bundle(self.stage3_pkl_path, as_ext=False) # [0,10,20,...,2760], 277
        self.num_frames = len(self.keyframes_idx)
        s3_ext_lj_to_li_s = [pose.matrix() for pose in s3_Pose3_lj_to_li_list]
        self.s3_ext_li_to_l0_s = utils.concat_cj_to_ci_s(s3_ext_lj_to_li_s)
        self.s3_dws = utils.get_dws_from_cam_to_world_s(self.s3_ext_li_to_l0_s)
        # kitti
        kitti_dws = kitti.read_dws(self.keyframes_idx)
        self.sd_list = [(self.s3_dws, 's3', 'red'), (kitti_dws, 'kitti', 'green')]
        # more init
        self.endframe = self.keyframes_idx[-1]
        self.init_dicts_and_dets_arr(s3_Pose3_lj_to_li_list, self.s3_cov_ln_cond_li_dict)        
        self.errors_before, self.errors_after = [], []
        self.k, self.ext_id, self.ext_l_to_r = kitti.read_cameras() # k=(3,4) ext_l0/r0 (4,4)
        self.gt_k = g_utils.get_gt_k(self.k, self.ext_l_to_r)
        print(f'INLIERS_PERC_LC_THRESH={INLIERS_PERC_LC_THRESH}')

        self.start_time = time.time()
        self.stats = [self.stage3_pkl_path, f'MAHAL_THRESH={MAHAL_THRESH}, INLIERS_PERC_LC_THRESH={INLIERS_PERC_LC_THRESH}']
        self.save = True

    def init_dicts_and_dets_arr(self, s3_Pose3_lj_to_li_list, s3_cov_lj_cond_li_s):
        num_frames = len(self.keyframes_idx) # 277
        self.from_to_Pose3_dict = defaultdict(dict)
        self.cov_ln_cond_li_dict = copy.deepcopy(s3_cov_lj_cond_li_s)
        self.det_ln_cond_li_arr = np.zeros( (num_frames, num_frames) )
        self.from_to_Pose3_dict[0][0] = s3_Pose3_lj_to_li_list[0]
        for j in range(1, num_frames): #[1,..,276]
            i = j-1
            self.from_to_Pose3_dict[j][i] = s3_Pose3_lj_to_li_list[j]
            self.det_ln_cond_li_arr[j,i] = np.linalg.det( self.cov_ln_cond_li_dict[j][i] )
    
        self.det_ln_cond_li_arr = csr_matrix(self.det_ln_cond_li_arr)
    
    def check_add_edges(self, i, n, prev_Pose3_ln_to_li, prev_cov_ln_cond_li, do_lc=True): # e.g. 57, 132
        # TODO Make this function static
        # consensus match        
        i_kf, n_kf = self.keyframes_idx[i], self.keyframes_idx[n]
        kp_li, kp_ri, pc_lr_i_in_li, kp_ln, kp_rn, pc_lr_n_in_ln, matches_li_ln = features.match_two_pairs(i_kf, n_kf)
        kp_li_m, kp_ri_m, pc_n_m = features.filter_with_matches(matches_li_ln, [kp_li, kp_ri], [pc_lr_n_in_ln])
        pnp_ext_ln_to_li, pnp_inliers_bool, _, _ = pnp.pnp_ransac(kp_li_m, kp_ri_m, pc_n_m, self.k, self.ext_l_to_r, max_iters=15, frame=f'{n_kf}, {i_kf}')
        pnp_perc_inliers = sum(pnp_inliers_bool) / len(matches_li_ln)
        kitti_ln_to_li = kitti.read_ln_to_li(i_kf, n_kf)
        rot_pnp_kitti, trans_pnp_kitti = utils.compare_ext_mat(pnp_ext_ln_to_li ,kitti_ln_to_li)
        # compare extrinsics
        prev_ext_ln_to_li = prev_Pose3_ln_to_li.matrix()
        rot_prev_kitti, trans_prev_kitti = utils.compare_ext_mat(prev_ext_ln_to_li, kitti_ln_to_li)
        added_edge = False
        if pnp_perc_inliers >= INLIERS_PERC_LC_THRESH and do_lc: # add edge
            msg = f'adding edge between {i_kf}, {n_kf}'
            self.stats.append(msg); print(msg)
            
            # prep small bundelon
            consistent_matches_li_ln = list(compress(matches_li_ln, pnp_inliers_bool))
            tracks_db = tracks.Tracks_DB()
            tracks_db.add_frame(consistent_matches_li_ln, i_kf, n_kf, kp_li, kp_ri, kp_ln, kp_rn, pc_lr_i_in_li, pc_lr_n_in_ln)
            frames_idx = [i_kf, n_kf]
            
            # do small bundelon
            values, error_before, error_after, bundle_marginals = bundle.do_single_bundle(frames_idx, [self.ext_id, pnp_ext_ln_to_li], tracks_db, self.gt_k)
            msg = f'bundleon [{i_kf},{n_kf}]: error before: {error_before:.1f}, after: {error_after:.1f}'
            self.stats.append(msg); print(msg)
            # bundle result
            Pose3_li_to_l0 = values.atPose3( X(i_kf) ); Pose3_ln_to_l0 = values.atPose3( X(n_kf) )
            Pose3_ln_to_li = Pose3_li_to_l0.between(Pose3_ln_to_l0) # bundle result
            cov_ln_cond_li = g_utils.extract_cov_ln_cond_li_from_marginals(bundle_marginals, i_kf, n_kf)
            cov_li_cond_ln = g_utils.extract_cov_ln_cond_li_from_marginals(bundle_marginals, n_kf, i_kf)
            det_ln_cond_li = np.linalg.det(cov_ln_cond_li)
            # update table
            self.from_to_Pose3_dict[n][i] = Pose3_ln_to_li
            self.cov_ln_cond_li_dict[n][i] = cov_ln_cond_li
            self.cov_ln_cond_li_dict[i][n] = cov_li_cond_ln
            self.det_ln_cond_li_arr[n,i] = det_ln_cond_li
            added_edge = True
            # compare new stuff
            # compare covariance
            prev_det = np.linalg.det(prev_cov_ln_cond_li); print('det prev cov_ln_cond_li', prev_det); print('det  new cov_ln_cond_li',  det_ln_cond_li)
            # compare extrinsics
            after_bundle_ext_ln_to_li = Pose3_ln_to_li.matrix()

            rot_after_bund_kitti, trans_after_bund_kitti = utils.compare_ext_mat(after_bundle_ext_ln_to_li, kitti_ln_to_li); rot_prev_pnp, trans_prev_pnp = utils.compare_ext_mat(prev_ext_ln_to_li, pnp_ext_ln_to_li)
            rot_pnp_after_bund, trans_pnp_after_bund = utils.compare_ext_mat(pnp_ext_ln_to_li, after_bundle_ext_ln_to_li); rot_prev_after_bund, trans_prev_after_bund = utils.compare_ext_mat(prev_ext_ln_to_li, after_bundle_ext_ln_to_li)
            print(f'prev_kitti:  rot = {rot_prev_kitti:.2f} deg, trans = {trans_prev_kitti:.2f} m'); print(f'bund_kitti:  rot = {rot_after_bund_kitti:.2f} deg, trans = {trans_after_bund_kitti:.2f} m')
            print(f'prev_pnp:    rot = {rot_prev_pnp:.2f} deg, trans = {trans_prev_pnp:.2f} m'); print(f'pnp_kitti:   rot = {rot_pnp_kitti:.2f} deg, trans = {trans_pnp_kitti:.2f} m')
            print(f'pnp_bundle:  rot = {rot_pnp_after_bund:.2f} deg, trans = {trans_pnp_after_bund:.2f} m'); print(f'prev_bundle: rot = {rot_prev_after_bund:.2f} deg, trans = {trans_prev_after_bund:.2f} m'); print()

        return added_edge, pnp_perc_inliers, rot_prev_kitti - rot_pnp_kitti, trans_prev_kitti - trans_pnp_kitti
    
    def update_after_pose_graph(self, values, marginals):
        # update Poses
        for n in self.from_to_Pose3_dict:
            n_kf = self.keyframes_idx[n]
            for i in self.from_to_Pose3_dict[n].keys():
                i_kf = self.keyframes_idx[i]
                self.from_to_Pose3_dict[n][i] = values.atPose3(X(i_kf)).between( values.atPose3(X(n_kf)) )
        
        # update cov and det_dits
        # self.det_ln_cond_li_arr = np.zeros((self.num_frames, self.num_frames))
        # for n in self.cov_ln_cond_li_dict:
        #     n_kf = self.keyframes_idx[n]
        #     for i in self.cov_ln_cond_li_dict[n].keys():
        #         i_kf = self.keyframes_idx[i]
        #         cov_ln_cond_li = g_utils.extract_cov_ln_cond_li_from_marginals(marginals, i_kf, n_kf)
        #         self.cov_ln_cond_li_dict[n][i] = cov_ln_cond_li
        #         if i < n: # update det
        #             det_cov_ln_cond_li = np.linalg.det(cov_ln_cond_li)
        #             self.det_ln_cond_li_arr[n,i] = det_cov_ln_cond_li
        # self.det_ln_cond_li_arr = csr_matrix(self.det_ln_cond_li_arr)
        
    def main(self, do_lc=True):
        print(f'do_lc={do_lc}')
        llsd_mahal, llsd_inliers, llsd_dets = [], [], []
        n = 40
        det_dists_ln_cond_li, pred_to_n = dijkstra(self.det_ln_cond_li_arr, directed=False, return_predecessors=True)
        # TODO change lines below when doing multiple loop closues
        # while n < 250 and self.num_of_lc < 7: # n in [1,2,...,276]
        while n < len(self.keyframes_idx): # n in [1,2,...,276]
            n_kf = self.keyframes_idx[n] # 200
            lc_edges_idx = []
            entered_pnp = False
            mahal_dists, inliers_percs, rot_diffs, trans_diffs = [], [], [], []
            # det_dists_ln_cond_li, pred_to_n = dijkstra(self.det_ln_cond_li_arr[(n+1):, (n+1):], directed=False, indices=n, return_predecessors=True)
            # det_dists_ln_cond_li, pred_to_n = dijkstra(self.det_ln_cond_li_arr, directed=False, indices=n, return_predecessors=True)
            llsd_dets.append([{'x':self.keyframes_idx, 'y':det_dists_ln_cond_li[n], 'name':'det_dist', 'mode':'markers+lines', 'cur_frame':n_kf}])
            if n % 20 == 0: print(f"main() {n_kf}/{self.keyframes_idx[-1]}")
            for i in range(n-30):
                # Try to loop closure
                i_kf = self.keyframes_idx[i]
                inliers_perc, rot_diff, trans_diff = 0,0,0
                Pose3_ln_to_li, cov_ln_cond_li, simp_path = shortest_path.Pose3_and_cov_ln_to_li_from_pred(i, n, self.from_to_Pose3_dict, self.cov_ln_cond_li_dict, pred_to_n[n])
                mahal_dist = g_utils.comp_mahal_dist(Pose3_ln_to_li, cov_ln_cond_li); mahal_dists.append(mahal_dist)
                if mahal_dist < MAHAL_THRESH and n-i > 20:
                    entered_pnp = True
                    added_edge, inliers_perc, rot_diff, trans_diff = self.check_add_edges(i, n, Pose3_ln_to_li, cov_ln_cond_li, do_lc=do_lc)
                    if added_edge: 
                        lc_edges_idx.append(i_kf)
                inliers_percs.append(inliers_perc); rot_diffs.append(rot_diff); trans_diffs.append(trans_diff)
            if len(mahal_dists)>3: llsd_mahal.append([{'x':self.keyframes_idx, 'y': mahal_dists, 'name':'mahal_dist', 'mode':'markers+lines', 'cur_frame':n_kf}])
            if entered_pnp and len(inliers_percs) > 3:
                llsd_inliers.append([{'x':self.keyframes_idx, 'y':inliers_percs, 'name':'inliers', 'mode':"markers+lines",
                            'cur_frame':n_kf, 'rot_diffs':rot_diffs, 'trans_diffs':trans_diffs}])        
            if lc_edges_idx and do_lc:
                # DO LOOP CLOSURE
                self.num_of_lc += 1
                self.cur_lc_dir = os.path.join(self.out_path, f'stage5_{self.num_of_lc}')
                utils.make_dir_if_needed(self.cur_lc_dir)
                # update table
                msg = f'doing LC on frame={n_kf} with edges {lc_edges_idx}'
                self.lc_msg = f'lc_{self.num_of_lc}_frame_{n_kf}_with_edges_{lc_edges_idx}'
                self.stats.append(msg); print(msg)
                # build pose graph with dijkstark
                
                graph, initialEstimate = self.build_pose_graph_with_dijk()
                pose_values, pose_marginals, error_before, error_after = pose_graph.optimize(graph, initialEstimate)
                msg = f'Pose graph {self.lc_msg} : error before: {error_before:.1f}, after: {error_after:.1f} \n'
                self.stats.append(msg); print(msg); self.errors_before.append(error_before); self.errors_after.append(error_after)
                self.update_after_pose_graph(pose_values, pose_marginals)
                det_dists_ln_cond_li, pred_to_n = self.output_results()
                n+=4
                print(f'jumping to frame={self.keyframes_idx[n+1]}')
            n+=1
        my_plot.pose_graph_llsd(self.keyframes_idx, llsd_inliers, llsd_mahal, llsd_dets, title="in_run", plot_dir=self.stage5_dir)
        self.output_after_main()
    
    def build_pose_graph_with_dijk(self):
        # BUILD POSE GRAPH WITH DIJKSTRA!!
        startframe = self.keyframes_idx[0]
       
        dijk_ext_li_to_l0_s, dijk_cov_li_on_l0_s, dijk_det_li_on_l0_s, _, _ = self.compute_ext_cov_li_to_l0_s_from_dijk(title=f"before_pose_and_update_lc_{self.num_of_lc}")
        graph = gtsam.NonlinearFactorGraph()

        # Create initial estimate - ci_to_c0
        initialEstimate = gtsam.Values()
        for i, dijk_li_to_l0 in enumerate (dijk_ext_li_to_l0_s): # [0,..,276]
            i_kf = self.keyframes_idx[i]
            initialEstimate.insert( X(i_kf), g_utils.to_Pose3(dijk_li_to_l0) )

        # add prior factor to graph
        pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1*pi/180, 1*pi/180, 1*pi/180, 0.3, 0.3, 0.3])) # (1 deg, input must be radian), 0.3 
        priorFactor = gtsam.PriorFactorPose3( X(startframe), Pose3() , pose_noise_model)
        graph.add(priorFactor)

        # add between factors to graph
        for n in range(1, self.num_frames):
            n_kf = self.keyframes_idx[n]
            for i, Pose3_n_to_i in self.from_to_Pose3_dict[n].items():
                i_kf = self.keyframes_idx[i]
                cov_cn_cond_ci = self.cov_ln_cond_li_dict[n][i]
                cov_ci_cond_c0 = dijk_cov_li_on_l0_s[i]
                cov_cn_cond_c0 = cov_cn_cond_ci + cov_ci_cond_c0
                noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_cn_cond_c0)
                factor = gtsam.BetweenFactorPose3( X(i_kf), X(n_kf) , g_utils.to_Pose3(Pose3_n_to_i), noise_model)
                graph.add(factor)
        
        return graph, initialEstimate
                
    def plot_mahal_and_inliers_perc(self, plot_dir, title):
        llsd_mahal, llsd_inliers = [], []
        llsd_dets = []
        det_dists_ln_cond_li, pred_to_n = dijkstra(self.det_ln_cond_li_arr, directed=False, return_predecessors=True)
        n = 40
        # while n < 150:
        while n < len(self.keyframes_idx): # n in [1,2,...,276]
            n_kf = self.keyframes_idx[n] # 20
            entered_pnp = False
            mahal_dists, inliers_percs, rot_diffs, trans_diffs = [], [], [], []
            # compute cov and pred
            llsd_dets.append([{'x':self.keyframes_idx, 'y':det_dists_ln_cond_li[n], 'name':'det_dist', 'mode':'markers+lines', 'cur_frame':n_kf}])
            if n % 20 == 0: print(f"plot_mahal_and_inliers() {title} {n_kf}/{self.keyframes_idx[-1]}")
            for i in range(n-30):
                inliers_perc, rot_diff, trans_diff = 0,0,0
                i_kf = self.keyframes_idx[i]
                Pose3_ln_to_li, cov_ln_cond_li, simp_path = shortest_path.Pose3_and_cov_ln_to_li_from_pred(i, n, self.from_to_Pose3_dict, self.cov_ln_cond_li_dict, pred_to_n[n])
                mahal_dist = g_utils.comp_mahal_dist(Pose3_ln_to_li, cov_ln_cond_li); mahal_dists.append(mahal_dist)
                if mahal_dist < MAHAL_THRESH:
                    entered_pnp = True
                    _ , inliers_perc, rot_diff, trans_diff = self.check_add_edges(i, n, Pose3_ln_to_li, cov_ln_cond_li, do_lc=False)
                inliers_percs.append(inliers_perc); rot_diffs.append(rot_diff); trans_diffs.append(trans_diff)
            if len(mahal_dists) > 5:
                llsd_mahal.append([{'x':self.keyframes_idx, 'y': mahal_dists, 'name':'mahal_dist', 'mode':'markers+lines', 'cur_frame':n_kf}])
            if entered_pnp and len(inliers_percs) > 3:
                llsd_inliers.append([{'x':self.keyframes_idx, 'y':inliers_percs, 'name':'inliers', 'mode':"markers+lines",
                            'cur_frame':n_kf, 'rot_diffs':rot_diffs, 'trans_diffs':trans_diffs}])        
            n+=1
        # plot
        my_plot.pose_graph_llsd(self.keyframes_idx, llsd_inliers, llsd_mahal, llsd_dets, title=title, plot_dir=plot_dir) 
    
    def compute_ext_cov_li_to_l0_s_from_dijk(self, title):
        num_frames = len(self.keyframes_idx) # 277
        startframe = self.keyframes_idx[0]
        det_dists_ln_cond_li, pred_to_n = dijkstra(self.det_ln_cond_li_arr, directed=False, return_predecessors=True)
        # get ext_li_to_l0_s and cov_li_cond_l0_s with dijkstra
        dijk_ext_li_to_l0_s = [np.diag([1,1,1,1])]
        dijk_cov_li_on_l0_s = [np.zeros((6,6))]
        dijk_det_li_on_l0_s, num_steps_to_zero = [0], [0]
        for n in range(1, num_frames): # [1,...,276]
            # find ln_to_l0_s
            Pose3_ln_to_l0, cov_ln_cond_l0, simp_path = shortest_path.Pose3_and_cov_ln_to_li_from_pred(0, n, self.from_to_Pose3_dict, self.cov_ln_cond_li_dict,
                                                        pred_to_n[n])
            dijk_ext_li_to_l0_s.append(Pose3_ln_to_l0.matrix())
            dijk_cov_li_on_l0_s.append(cov_ln_cond_l0)
            dijk_det_li_on_l0_s.append(np.linalg.det(cov_ln_cond_l0))
            num_steps_to_zero.append(len(simp_path)-1 )

        dijk_dws = utils.get_dws_from_cam_to_world_s(dijk_ext_li_to_l0_s)
        dijk_sd = (dijk_dws, f'dijk_dws_{self.num_of_lc}', 'pink')
        my_plot.plotly_scatter(x=self.keyframes_idx, y=dijk_det_li_on_l0_s, yaxis='det of cov', plot_dir=self.cur_lc_dir,
                               title=f"dijk_det_cov_li_cond_on_l0_s_{title}", save=True, plot=False)
        my_plot.plotly_scatter(x=self.keyframes_idx, y=num_steps_to_zero, yaxis='num of steps to zero', plot_dir=self.cur_lc_dir,
                               title=f"dijk_length_path_to_zero_{title}", save=True, plot=False)
        tmp_sd = self.sd_list + [dijk_sd]
        my_plot.plotly_3D_cams(tmp_sd, title=f"dijk_{title}", plot_dir=self.cur_lc_dir, frames_idx=self.keyframes_idx, save=True, plot=False)
        
        return dijk_ext_li_to_l0_s, dijk_cov_li_on_l0_s, dijk_det_li_on_l0_s, det_dists_ln_cond_li, pred_to_n
    
    def output_results(self):
        # Create new stage5 folder
        self.cur_lc_dir = os.path.join(self.out_path, f'stage5_{self.num_of_lc}')
        utils.make_dir_if_needed(self.cur_lc_dir)
        # serialzie stage 5
        g_utils.serialize_stage5( self.cur_lc_dir, self.from_to_Pose3_dict, self.cov_ln_cond_li_dict, self.det_ln_cond_li_arr,
                                 self.keyframes_idx, title=f'stage5_{self.num_of_lc}')
        
        
        # plot mahal and inliers
        self.plot_mahal_and_inliers_perc(plot_dir=self.cur_lc_dir, title=f'stage5_{self.lc_msg}')
        ext_li_to_l0_s, cov_li_on_l0_s, det_li_on_l0_s, det_dists_ln_cond_li, pred_to_n = \
                     self.compute_ext_cov_li_to_l0_s_from_dijk(f"after_optimize_and_update_{self.lc_msg}")

        new_dws = utils.get_dws_from_cam_to_world_s(ext_li_to_l0_s)
        new_sd = (new_dws, f'lc_{self.num_of_lc}', utils.get_color(self.num_of_lc))
        self.sd_list.append(new_sd)

        rots_total_error_abs, trans_total_error_abs = results.output_results(self.out_path, ext_li_to_l0_s, self.keyframes_idx, f"stage_5_{self.num_of_lc}",
                                                     self.start_time, save=self.save, plot=False)
        # Rename stage5 folder
        new_stage5_dir = os.path.join(self.out_path, f'stage5_{self.num_of_lc}_{rots_total_error_abs:.1f}_{trans_total_error_abs:.1f}')
        utils.clear_and_make_dir(new_stage5_dir)
        os.rename(self.cur_lc_dir, new_stage5_dir)
        self.cur_lc_dir = new_stage5_dir

        return det_dists_ln_cond_li, pred_to_n

        
    def output_after_main(self):
        # write stats
        if self.stats and self.save:
            with open (os.path.join(self.stage5_dir,f'stats_stage5.txt'), 'w') as f:
                f.writelines('\n'.join(self.stats))
        if self.errors_before and self.errors_after:
                err_dict = {'before':self.errors_before, 'after':self.errors_after}
                my_plot.plotly_scatters(err_dict,title="stage5_pose_graph_errors", plot_dir=self.stage5_dir, yaxis="error", xaxis="LC num", save=True, plot=False)

if __name__=="__main__":
    stg3_pkl_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-10-13-49_0_2760/stage3_40.8_29.9/stage3_ext_lj_to_li_s_cond_covs_2760.pkl'
    lc = LoopClosure(stg3_pkl_path)
    lc.main(do_lc=True)
    print('pose_graph finished')

