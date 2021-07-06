import numpy as np
from numpy import pi as pi
import gtsam
from gtsam.symbol_shorthand import X, P
import os, time
from itertools import compress

import kitti, tracks, utils, my_plot, results, features, bundle, pnp
import gtsam_utils as g_utils

MAHAL_THRESH = 1
INLIERS_PERC_LC_THRESH = 0.4

class PoseGraph:
    def __init__(self, pkl_path):
        # Folders
        pkl_os_path = utils.path_to_current_os(pkl_path)
        self.stage3_dir, _, _ = utils.dir_name_ext(pkl_os_path)
        self.pkl_path = pkl_os_path
        self.out_path = os.path.dirname(self.stage3_dir)
        self.stage5_dir = os.path.join(self.out_path, 'stage5')
        utils.make_dir_if_needed(self.stage5_dir)
        # Data
        self.Pose3_li_to_l0_s, self.cov_lj_cond_li_keyframes, self.keyframes_idx = g_utils.unserialize_bundle(pkl_os_path) # [0,10,20,...,2760], 277
        self.endframe = self.keyframes_idx[-1]
        self.cov_li_cond_l0_cumsum = g_utils.cumsum_mats(self.cov_lj_cond_li_keyframes) 
        self.cov_li_cond_l0_cumsum.insert(0, np.zeros((6,6))) # (277,) a[i]= Sigma_i|0
        self.errors_before, self.errors_after = [], []
        self.k, self.ext_id, self.ext_l_to_r = kitti.read_cameras() # k=(3,4) ext_l0/r0 (4,4)
        self.gt_k = g_utils.get_gt_k(self.k, self.ext_l_to_r)
        
        self.num_of_lc = 0
        self.start_time = time.time()
        self.stats = [pkl_os_path, f'MAHAL_THRESH={MAHAL_THRESH}, INLIERS_PERC_LC_THRESH={INLIERS_PERC_LC_THRESH}']
        self.save = True
    
    def cov_cn_cond_on_ci(self, i, n): # e.g. 2, 5
        cov_cn_cond_on_ci = self.cov_li_cond_l0_cumsum[n] - self.cov_li_cond_l0_cumsum[i] # (6,6)
        return cov_cn_cond_on_ci 
    
    def consensus_match(self, i_kf, n_kf):
        kp_li, kp_ri, pc_lr_i_in_li, kp_ln, kp_rn, matches_li_ln = features.match_two_pairs(i_kf, n_kf)
        pc_lr_i_in_li_m, kp_ln_m, kp_rn_m = features.filter_with_matches(matches_li_ln, [pc_lr_i_in_li], [kp_ln, kp_rn])
        ext_li_to_ln, ext_inliers_bool, _, _ = pnp.pnp_ransac(kp_ln_m, kp_rn_m, pc_lr_i_in_li_m, self.k,
                                                              self.ext_l_to_r, max_iters=15, frame=f'({i_kf},{n_kf})')  # (4,4), (len(matches)
        perc_inliers = sum(ext_inliers_bool) / len(matches_li_ln)
        return perc_inliers
    
    def check_add_edges(self, i_kf, n_kf):
        # consensus match
        kp_li, kp_ri, pc_lr_i_in_li, kp_ln, kp_rn, matches_li_ln = features.match_two_pairs(i_kf, n_kf)
        kp_ln_m, kp_rn_m, pc_lr_i_in_li_m = features.filter_train(matches_li_ln, [kp_ln, kp_rn, pc_lr_i_in_li])
        ext_li_to_ln, ext_inliers_bool, _, _ = pnp.pnp_ransac(kp_ln_m, kp_rn_m, pc_lr_i_in_li_m, self.k,
                                                              self.ext_l_to_r, max_iters=15, frame=f'{i_kf},{n_kf}')  # (4,4), (len(matches)
        perc_inliers = sum(ext_inliers_bool) / len(matches_li_ln)
        added_edge = False
        if perc_inliers >= INLIERS_PERC_LC_THRESH: # add edge
            added_edge = True
            msg = f'adding edge between {i_kf}, {n_kf}'
            self.stats.append(msg); print(msg)
            
            # prep small bundelon
            ext_ln_to_li = utils.inv_extrinsics(ext_li_to_ln)
            # TODO maybe I can get ext_ln_to_li simply by sending different stuff to pnp? CHECK
            consistent_matches_li_ln = list(compress(matches_li_ln, ext_inliers_bool))
            tracks_db = tracks.Tracks_DB()
            tracks_db.add_frame(consistent_matches_li_ln, i_kf, n_kf,
                                kp_li, kp_ri, kp_ln, kp_rn, pc_lr_i_in_li, None)
            frames_idx = [i_kf, n_kf]
            
            # do small bundelon
            values, error_before, error_after, bundle_marginals = bundle.single_bundle(frames_idx, [self.ext_id, ext_ln_to_li], tracks_db, self.gt_k)
            msg= f'bundle [{i_kf},{n_kf}]: error before: {error_before:.1f}, after: {error_after:.1f}'
            self.stats.append(msg); print(msg)
            
            # add betweenFactor to pose graph
            cov_cn_cond_on_ci = g_utils.cov_ln_cond_on_li(bundle_marginals, n_kf, i_kf)
            Pose3_ln_to_li = values.atPose3( X(n_kf) )
            noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_cn_cond_on_ci)
            factor = gtsam.BetweenFactorPose3( X(i_kf), X(n_kf) , Pose3_ln_to_li, noise_model)
            self.graph.add(factor)

        return ext_li_to_ln, added_edge
      
    def comp_mahal_dist(self, i, n): # 6, 2
        i_kf, n_kf = self.keyframes_idx[i], self.keyframes_idx[n]
        Pose3_cn_to_c0 = self.Pose3_li_to_l0_s.atPose3( X(n_kf) )
        Pose3_ci_to_c0 = self.Pose3_li_to_l0_s.atPose3( X(i_kf) )
        Pose3_cn_to_ci_mine = g_utils.Pose3_cn_to_ci(Pose3_cn_to_c0, Pose3_ci_to_c0)
        Pose3_cn_to_ci_gt = Pose3_ci_to_c0.between(Pose3_cn_to_c0)
        eqa = np.allclose(Pose3_cn_to_ci_mine.matrix(), Pose3_cn_to_ci_gt.matrix())
        if not eqa:
            print(f"not equal, i={i_kf}, n={n_kf}")
        t2v = g_utils.t2v(Pose3_cn_to_ci_mine)
        cov_cn_cond_on_ci = self.cov_cn_cond_on_ci(i, n) # (6,6)
        mahal_dist = t2v.T @ cov_cn_cond_on_ci @ t2v
        return mahal_dist

    def shortest_path(self, n_kf_idx):
        # TODO implement shortest path: graph is keyframe poses, edge weights the determinant of the cov matrices
        # TODO no need to check for previous 10 keyframes
        pass
    
    def optimize(self):
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initialEstimate)
        self.errors_before.append(optimizer.error())
        values = optimizer.optimize()
        self.errors_after.append(optimizer.error())
        msg = f'Pose graph [{0}-{self.endframe}]: error before: {self.errors_before[-1]:.1f}, after: {self.errors_after[-1]:.1f}'
        self.stats.append(msg); print(msg)
        pose_marginals = gtsam.Marginals(self.graph, values)
        self.Pose3_li_to_l0_s = gtsam.utilities.allPose3s(values)
        self.cov_lj_cond_li_keyframes, self.cov_li_cond_l0_cumsum = g_utils.cov_lj_cond_on_li_s(pose_marginals, self.keyframes_idx)
    
    def main(self):
        self.graph, self.initialEstimate = build_pose_graph(self.keyframes_idx, self.Pose3_li_to_l0_s, self.cov_lj_cond_li_keyframes)
        n = 20
        while n < len(self.keyframes_idx): # n in [1,2,...,276]
            n_kf = self.keyframes_idx[n] # 200
            edges_added = []
            for i in range(n):
                i_kf = self.keyframes_idx[i]
                mahal_dist = self.comp_mahal_dist(i, n)
                if mahal_dist < MAHAL_THRESH:
                    ext_li_to_ln, edge_added = self.check_add_edges(i_kf, n_kf)
                    if edge_added: edge_added.append(i_kf)
            if edges_added:
                print(f'did LC on frame={n_kf} with edges [{edges_added}]')
                self.num_of_lc += 1
                self.optimize()
                build_pose_graph(self.keyframes_idx, self.Pose3_li_to_l0_s, self.cov_lj_cond_li_keyframes)    
                # j+=4
            n+=1        
        self.output_results()

    def plot_mahal_and_inliers_perc(self):
        llsd_mahal = []
        llsd_inliers = []
        for n in range(2, 101):
        # for n in range(2, len(self.keyframes_idx)):
            mahal_dists, frames_inliers_perc = [], []
            n_kf = self.keyframes_idx[n]
            for i in range(n):
                i_kf = self.keyframes_idx[i]
                mahal_dist = self.comp_mahal_dist(i, n)
                mahal_dists.append(mahal_dist)
                if mahal_dist < MAHAL_THRESH and n-i > 20:
                    inliers_perc = self.consensus_match(i_kf, n_kf)
                    frames_inliers_perc.append( (i_kf, inliers_perc) )
            
            mahal_scat = {'x':self.keyframes_idx, 'y':mahal_dists, 'name':'mahal_dist', 'mode':'markers+lines', 'cur_frame':i_kf}
            llsd_mahal.append([mahal_scat])            
            if frames_inliers_perc:
                frames_inliers_perc = np.array(frames_inliers_perc).T
                inliers_scatter = {'x':frames_inliers_perc[0], 'y':frames_inliers_perc[1] * 100, 'name':'inliers', 'mode':"markers+lines", 'cur_frame':i_kf}
                llsd_inliers.append([inliers_scatter])        
        
        if llsd_mahal: my_plot.frames_slider_plot(llsd_mahal, title="mahal_dist", plot_dir=self.stage5_dir, yaxis="mahal dist")
        if llsd_inliers: my_plot.frames_slider_plot(llsd_inliers, title="inliers_perc", plot_dir=self.stage5_dir, yaxis=r"% inliers")
    
    def output_results(self):
        ext_li_to_l0_s = g_utils.ext_ci_to_c0_s_from_values(self.Pose3_li_to_l0_s)
        rots_total_error, trans_total_error = results.output_results(self.out_path, ext_li_to_l0_s,
                            self.keyframes_idx, f"stage_5_{self.num_of_lc}", self.start_time, relative=False, plot=False, save=self.save)
        
        # rename stage5 folder
        new_stage5_dir = self.stage5_dir + f'{self.num_of_lc}_{trans_total_error:.1f}_{rots_total_error:.1f}'
        utils.clear_and_make_dir(new_stage5_dir)
        os.rename(self.stage5_dir, new_stage5_dir)
        self.stage5_dir = new_stage5_dir   
        
        # plot mahal and inliers
        self.plot_mahal_and_inliers_perc()
        
        # write stats
        with open (os.path.join(self.stage5_dir,'stats_stage5.txt'), 'w') as f:
            f.writelines('\n'.join(self.stats))
        
        # graph before_after optimization errors
        idx = np.arange(len(self.errors_before))
        if self.errors_before and self.errors_after:
            my_plot.plt_bundle_errors(self.errors_before, self.errors_after, idx, "stage5_pose_graphs", self.stage5_dir, xlabel="LC num", save=self.save, plot=False)

        # serialzie Pose3 and marginals
        g_utils.serialize_bundle(self.stage5_dir, self.Pose3_li_to_l0_s, self.cov_lj_cond_li_keyframes, self.keyframes_idx, "stage5")

def build_pose_graph(keyframes_idx, Pose3_li_to_l0_s, cov_lj_cond_li):
    graph = gtsam.NonlinearFactorGraph()

    # Create initial estimate
    initialEstimate = gtsam.Values()
    for keyframe_idx in keyframes_idx:
        Pose3_li_to_l0 = Pose3_li_to_l0_s.atPose3( X(keyframe_idx) )
        initialEstimate.insert( X(keyframe_idx), Pose3_li_to_l0 )
    
    # add prior factor to graph
    pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1*pi/180, 1*pi/180, 1*pi/180, 0.3, 0.3, 0.3])) # (1 deg, input must be radian), 0.3 meter
    init_pose = Pose3_li_to_l0_s.atPose3( X(keyframes_idx[0]) )
    priorFactor = gtsam.PriorFactorPose3( X(keyframes_idx[0]), init_pose, pose_noise_model)
    graph.add(priorFactor)

    # add between factors to graph
    for j in range(1, len(keyframes_idx)): # [1,2,...,276] represent [10, 20, .. 2760]
        i = j-1 # 0
        i_kf, j_kf = keyframes_idx[i], keyframes_idx[j] # 0, 10
        pose_ci = Pose3_li_to_l0_s.atPose3( X(i_kf) )
        pose_cj = Pose3_li_to_l0_s.atPose3( X(j_kf) )
        cj_to_ci_pose = pose_ci.between(pose_cj)
        cj_cond_ci_cov = cov_lj_cond_li[i] # ndarray (6,6)
        noise_model = gtsam.noiseModel.Gaussian.Covariance(cj_cond_ci_cov)
        factor = gtsam.BetweenFactorPose3( X(i_kf), X(j_kf) , cj_to_ci_pose, noise_model)
        graph.add(factor)

    return graph, initialEstimate

if __name__=="__main__":
    # pkl_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/06-13-16-50_mine_global_50/stage3_0.8_0.7/Pose3_marginals_50.pkl'
    # pkl_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/06-13-14-30_mine_global_200/stage3_14.9_5.9/Pose3_marginals_200.pkl'
    # pkl_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/06-10-15-12_mine_global_2760/stage3_433.1_75.9/stage3_Pose3_marginals_2760.pkl'
    pkl_path = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-02-01-24_mine_global_2760/stage3_420.7_74.4/Pose3_marginals_2760.pkl'
    pg = PoseGraph(pkl_path)
    pg.output_results()
    # pg.main()
    print('pose_graph finished')

