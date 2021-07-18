import numpy as np
from numpy import pi
import gtsam
from gtsam import Pose3, utils as g_utils
from gtsam.symbol_shorthand import X

from collections import defaultdict


def extract_ext_ci_to_c0_s_from_values(values, keyframes_idx):
    ext_ci_to_c0_s = [values.atPose3(X(i_kf)).matrix() for i_kf in keyframes_idx]
    return ext_ci_to_c0_s

def extract_Pose3_list_from_values(values, keyframes_idx):
    Pose3_list = [values.atPose3( X(i_kf) ) for i_kf in keyframes_idx]
    return Pose3_list
    
def extract_ext_cond_from_values_marginals(values, marginals, keyframes_idx):
    ext_ci_to_c0_s = []
    for i_kf in keyframes_idx:
        Pose3_ci_to_c0 = values.atPose3( X(i_kf))
        ext_ci_to_c0_s.append(Pose3_ci_to_c0.matrix())

    cov_cj_cond_ci_dict = defaultdict(dict)
    for j in range(1, len(keyframes_idx)):
        i_kf, j_kf = keyframes_idx[j-1], keyframes_idx[j]
        cov_cj_cond_ci = g_utils.extract_cov_ln_cond_li_from_marginals(marginals, i_kf, j_kf); 
        cov_cj_cond_ci_dict[j][j-1] = cov_cj_cond_ci
        cov_ci_cond_cj = g_utils.extract_cov_ln_cond_li_from_marginals(marginals, j_kf, i_kf);
        cov_cj_cond_ci_dict[j-1][j] = cov_ci_cond_cj
    return ext_ci_to_c0_s, cov_cj_cond_ci_dict



def build_pose_graph(keyframes_idx, ci_to_c0_s, from_cn_to_ci_dict, cov_cn_cond_ci_dict):
    if type(ci_to_c0_s) == type(gtsam.Values()):
        ci_to_c0_s = extract_Pose3_list_from_values(ci_to_c0_s)
    assert len(ci_to_c0_s)==len(keyframes_idx)
    graph = gtsam.NonlinearFactorGraph()
    startframe = keyframes_idx[0]

    # Create initial estimate
    initialEstimate = gtsam.Values()
    for i_kf, ci_to_c0 in zip(keyframes_idx, ci_to_c0_s):
        initialEstimate.insert( X(i_kf), g_utils.to_Pose3(ci_to_c0))
    
    # add prior factor to graph
    pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1*pi/180, 1*pi/180, 1*pi/180, 0.3, 0.3, 0.3])) # (1 deg, input must be radian), 0.3 
    priorFactor = gtsam.PriorFactorPose3( X(startframe), Pose3() , pose_noise_model)
    graph.add(priorFactor)

    # add between factors to graph
    # TODO !!! first time try compute between_pose using Pose3_li_to_l0_s. I should get zero error. Then try using from_to_Pose3_dict, should get higher error
    for n in range(1, len(keyframes_idx)): # [1,2,...,276] represent [10, 20, .. 2760]
        n_kf = keyframes_idx[n]
        for i, cn_to_ci in from_cn_to_ci_dict[n].items():
            i_kf = keyframes_idx[i]
            cov_cn_cond_ci = cov_cn_cond_ci_dict[n][i]
            noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_cn_cond_ci)
            factor = gtsam.BetweenFactorPose3( X(i_kf), X(n_kf) , g_utils.to_Pose3(cn_to_ci), noise_model)
            graph.add(factor)
    return graph, initialEstimate

def optimize(graph, initialEstimate):
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
    error_before = optimizer.error()
    values = optimizer.optimize()
    error_after = optimizer.error()
    marginals = gtsam.Marginals(graph, values)
    return values, marginals, error_before, error_after

