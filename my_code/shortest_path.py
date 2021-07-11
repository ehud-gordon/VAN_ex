import numpy as np
import gtsam
from gtsam import Pose3
import utils

def shortest_path_from_n_to_i(i, n, pred):
    """ if n=5 and i=0, returns [(1, 0), (3, 1), (4, 3), (5, 4)]
    :param pred: (n,) array from calling scipy.dijkstra(indices=n, return_predecessors=True)
    """
    assert i < len(pred) and n < len(pred)
    assert i!=n
    if i==n: return []
    res = []
    start=i
    while start != n:
        next = pred[start]
        res.append((next,start))
        start = next
    return res

def Pose3_and_cov_ln_to_li_from_pred(i, n, from_to_Pose3_dict, cov_ln_cond_li_dict, pred_to_n):
    path = shortest_path_from_n_to_i(i,n, pred_to_n) # list of tuples, from n=5 to =0 [(1,0), (3,1), (4,3), (5,4)]
    Pose3_ln_to_li = Pose3()
    cov_ln_cond_li = np.zeros((6,6))
    for k,j in path: # (1,0) (133, 132) take 133_to_132
        
        if j > k: # if we have k,j=(132, 133) we need to take INVERSE(133_to_132)
            # TODO and equally some correction for cov?
            cj_to_ck_Pose3 = from_to_Pose3_dict[j][k]
            ck_to_cj_Pose3 = cj_to_ck_Pose3.inverse()
            Pose3_ln_to_li = Pose3_ln_to_li.compose(ck_to_cj_Pose3)
            cov_ln_cond_li += cov_ln_cond_li_dict[k][j]
        else:
            ck_to_cj_Pose3 = from_to_Pose3_dict[k][j]
            Pose3_ln_to_li = Pose3_ln_to_li.compose(ck_to_cj_Pose3)
            cov_ln_cond_li += cov_ln_cond_li_dict[k][j]
    simp_path = [t[0] for t in path]
    simp_path.insert(0,path[0][1])

    return Pose3_ln_to_li, cov_ln_cond_li, simp_path
