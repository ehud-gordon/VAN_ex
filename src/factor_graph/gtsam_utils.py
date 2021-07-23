""" Collection of utilities for use with gtsam package """
import gtsam
import utils.geometry
from gtsam import KeyVector, Pose3
from gtsam.utils import plot as g_plot
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import defaultdict
from gtsam.symbol_shorthand import X
import os, pickle

import utils
from utils.sys_utils import und_title

def is_gt(pose):
    return gtsam.__name__ in str(type(pose))

def get_trans(pose):
    if is_gt(pose):
        return pose.translation()
    else:
        return pose[:3,-1]

def get_rot(pose):
    if is_gt(pose):
        return pose.rotation().matrix()
    else:
        return pose[:3,:3]

def rot_trans(pose):
    return get_rot(pose), get_trans(pose)

def get_dws(poses):
    trans_vecs = [get_trans(pose) for pose in poses]
    return np.array(trans_vecs).T

def t2v(pose3):
    rot_mat = pose3.rotation().matrix()    
    trans = pose3.translation()
    return utils.geometry.t2v(rot_mat, trans)

def get_stereo_camera(k, ext_l_to_r):
    """ Create stereo camera in format gtsam requires for GenericStereoFactor3D"""
    fx, skew, cx, _, fy, cy = k[0:2, 0:3].flatten()
    baseline = ext_l_to_r[0, 3]
    return gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, -baseline)


def conditional_covariance_from_Marginals(marginals, i, n):
    """ compute conditional covariance, Sigma n|i.

    :param marginals: gtsam.Marginals object
    :param i: index of frame/camera i
    :param n: index of frame/camera i
    :return: (6,6) ndarray,  Sigma n|i
    """
    keys = KeyVector([X(i), X(n)])
    n_cond_on_i_info = marginals.jointMarginalInformation(keys).at(X(n), X(n))
    n_cond_on_i_cov = np.linalg.inv(n_cond_on_i_info)
    return n_cond_on_i_cov


#### VISUALIZATION ######
def get_ellipsoid_trace(pose, P, name=""):
    """
    :param pose: cam to world. 
    :param P: (6,6) covariance matrix of this pose
    This code is copied from gtsam.utils.plot.
    """

    gRp, origin = rot_trans(pose)
    pPp = P[3:6, 3:6]
    gPp = gRp @ pPp @ gRp.T
    k = 11.82
    U, S, _ = np.linalg.svd(gPp)

    radii = k * np.sqrt(S)
    radii = radii
    rx, ry, rz = radii

    # generate data for "unrotated" ellipsoid
    xc, yc, zc = g_plot.ellipsoid(0, 0, 0, rx, ry, rz, 8)

    # rotate data with orientation matrix U and center c
    data = np.kron(U[:, 0:1], xc) + np.kron(U[:, 1:2], yc) + np.kron(U[:, 2:3], zc)
    n = data.shape[1]
    x = data[0:n, :] + origin[0]
    y = data[n:2*n, :] + origin[1]
    z = data[2*n:, :] + origin[2]

    ellipse_trace = go.Surface(x=x, y=z, z=y, opacity=0.5, showscale=False, showlegend=(name != ""), name=name)
    return ellipse_trace, x,y,z
