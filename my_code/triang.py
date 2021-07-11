import matplotlib.pyplot as plt
import numpy as np
import cv2

import os

import utils

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.5g" % x))

def relative_inliers(pc):
    """ :param pc: (3,n) """
    x_abs = np.abs(pc[0]); y_abs = np.abs(pc[1])
    x_crit = (x_abs <= 30)
    y_crit = (y_abs <= 30)
    z_crit1 = pc[2] < 200
    z_crit2 = pc[2] > 1
    z_crit = z_crit1 * z_crit2
    inliers = (x_crit * y_crit) * z_crit
    return inliers

def isolation_forest_inliers(pc, n_est=100, cont=0.01):
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(n_estimators=n_est, contamination=cont)
    inliers_is_forest = (clf.fit_predict(pc.T) > 0)
    return inliers_is_forest

def quantile_inliers(pc, q=0.99):
    pc_abs = np.abs(pc)
    x_quant, y_quant, z_quant = np.quantile(pc_abs, q=q, axis=1)
    x_crit = (pc_abs[0] <= x_quant)
    y_crit = (pc_abs[1] <= y_quant)
    z_crit = (pc_abs[2] <= z_quant)
    inliers = x_crit * y_crit * z_crit
    return inliers

def quant_forest(pc, q=0.99, n_est=100, cont=0.01, less_inliers=True):
    """ :param pc: (3,n) """
    inliers_quant = quantile_inliers(pc, q=q)
    inliers_is_for = isolation_forest_inliers(pc, n_est=n_est, cont=cont)
    if less_inliers:
        less_inliers = inliers_quant * inliers_is_for # AND (less inliers, more outliers)
        return less_inliers
    else:
        more_inliers = ~(~inliers_quant * ~inliers_is_for) # OR (more inliers, less outliers)
        return more_inliers

def triang(kpA, kpB, k, ext_WORLD_to_A, ext_WORLD_to_B):
    """
    get 3D, WORLD coordinates, via triangulation of matched pixels from two images
    :param kpA/B: pixels of matched points in imageA/B, (2,n)
    :param k: intrinsics  matrix shared by cameras A and B, (3,4)
    :param mA: extrinsics matrix from WORLD to CS_A (4,4)
    :param mB: extrinsics matrix from WORLD to CS_B (4,4)
    :return: (3,n) ndarray of world coordinates in WORLD CS (in kitti, this means left0)
    """
    assert kpA.shape == kpB.shape
    proj_WORLD_to_A = k @ ext_WORLD_to_A  # (3,4) # projection matrix
    proj_WORLD_to_B = k @ ext_WORLD_to_B  # (3,4)

    pc_4d = cv2.triangulatePoints(projMatr1=proj_WORLD_to_A, projMatr2=proj_WORLD_to_B, projPoints1=kpA, projPoints2=kpB)  # (4,n)
    pc_3d = pc_4d[0:3] / pc_4d[-1]  # (3,n)
    return pc_3d # points in WORLD CS

def triang_and_rel_filter(kpA, kpB, k, ext_WORLD_to_A, ext_WORLD_to_B, *nd_arrays):
    pc = triang(kpA, kpB, k, ext_WORLD_to_A, ext_WORLD_to_B)
    rel_filter = relative_inliers(pc)
    return utils.filt_np(rel_filter, kpA, kpB, pc, *nd_arrays)


#########################   Visualization utils  #########################
def vis_pc(pc, title="", save=False, inliers_bool=None, hist=False):
    """ :param pc: (3,num_of_matches), inhomogeneous"""
    assert pc.shape[0] in [3,4]
    if pc.shape[0] == 4:
        pc = pc[:3] / pc[-1,:]
    if hist:
        plt.figure(); plt.hist(pc[0], density=True, label="x")
        plt.title(f"x_{title}"); plt.ylabel('x_value'); plt.show()
        plt.figure(); plt.hist(pc[1], density=True, label="y")
        plt.title(f"y_{title}"); plt.ylabel('y_value'); plt.show()
        plt.figure(); plt.hist(pc[2], density=True, label="z")
        plt.title(f"z_{title}"); plt.ylabel('z_value'); plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if inliers_bool is not None:
        # ax.scatter(cameras[0,:],cameras[2,:], cameras[1,:],marker=(5,2), color="red", s=50)
        ax.scatter(pc[0, inliers_bool], pc[2, inliers_bool], pc[1, inliers_bool], color="blue", label="inliers")  # this isn't a mistake, plt's z axis is our Y axis
        ax.scatter(pc[0, ~inliers_bool], pc[2, ~inliers_bool], pc[1, ~inliers_bool], color="red", label="outliers")  # this isn't a mistake, plt's z axis is our Y axis
    else:
        ax.scatter(pc[0, :], pc[2, :], pc[1, :], color="blue") # this isn't a mistake, plt's z axis is our Y axis
    ax.set_title(title)
    xmin, ymin, zmin = np.min(pc, axis=1)
    xmax, ymax, zmax = np.max(pc, axis=1)
    ax.set_ylim([0, zmax + 1])  # not a mistake, plt's Y axis is our Z-Axis
    ax.set_xlim([xmin - 1, xmax + 1])
    ax.set_zlim([ymin - 1, ymax + 1]) # not a mistake, plt's z-axis is our Y-axis
    ax.invert_zaxis()  # not a mistake, - plt's z axis is our Y axis
    ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y') # not a mistake
    plt.legend()
    if save:
        path = utils.get_avail_path(os.path.join(utils.out_dir(), f'{title}_pc.png'))
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

def vis_triang(img, pc, pxls, title="", save=False):
    """ :param pc: (3,num_of_matches)
        :param pxls: (2,num_of_matches) """
    assert pc.shape[1] == pxls.shape[1]
    vis_pc(pc=pc,title=title, save=save)
    # plot pxls with txt indicating their 3d location
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    num_matches = pc.shape[1]
    rand_inds = np.random.randint(0, num_matches, size=20)
    for ind in rand_inds:
        x_w, y_w, z_w = pc[0:3, ind]
        x, y = pxls[:, ind]
        x = int(x); y = int(y)
        print(f'({x},{y:}) -> ({x_w:.1f},{y_w:.1f},{z_w:.1f})')
        img = cv2.circle(img, center=(x, y), radius=2, color=(0, 0, 0),thickness=1)
        img = cv2.putText(img, f'{x_w:.1f},{y_w:.1f},{z_w:.1f}', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                           color=(0, 255, 255), lineType=cv2.LINE_AA)
    print()
    utils.cv_disp_img(img, title=f"{title}_vis_tr",save=save)

if __name__=="__main__":
    import kitti, features, cv2, my_plot
    pass