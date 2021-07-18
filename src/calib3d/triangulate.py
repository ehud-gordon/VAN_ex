"""
Performs triangulation. I've also included my version,
but the code uses cv2.triangulatePoints(), for performance reasons.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import utils
import utils.sys_utils as sys_utils

def triangulate(pixels1, pixels2, k, ext_WORLD_to_1, ext_WORLD_to_2):
    """
    Get 3D, WORLD coordinates, via triangulation of matched pixels from two images
    :param pixels1: (2,n) ndarray of pixels in images 1 that're matched to pixels2
    :param pixels2: (2,n) ndarray of pixels in images 2 that're matched to pixels1
    :param k: intrinsics matrix shared by cameras 1 and 2, (3,4) ndarray
    :param ext_WORLD_to_1: extrinsics matrix from WORLD to coordinate-system 1, (4,4) ndarray
    :param ext_WORLD_to_2: extrinsics matrix from WORLD to coordinate-system 2, (4,4) ndarray
    :return: pc_3d: a (3,n) ndarray of the point cloud - landmarks in WORLD coordinate-system
    """
    assert pixels1.shape == pixels2.shape
    proj_WORLD_to_1 = k @ ext_WORLD_to_1  # (3,4) ndarray, projection matrix from world CS to CS1
    proj_WORLD_to_2 = k @ ext_WORLD_to_2  # (3,4) ndarray, projection matrix from world CS to CS2
    # get pc (point cloud)
    pc_4d = cv2.triangulatePoints(projMatr1=proj_WORLD_to_1, projMatr2=proj_WORLD_to_2, projPoints1=pixels1, projPoints2=pixels2)  # (4,n)
    pc_3d = pc_4d[0:3] / pc_4d[-1]  # (3,n)
    return pc_3d # points in WORLD CS

def my_triangulate(pixels1, pixels2, projMatr1, projMatr2):
    """
    Get 3D, WORLD coordinates, via triangulation of matched pixels from two images
    :param pixels1: (2,n) ndarray of pixels in images 1 that're matched to pixels2
    :param pixels2: (2,n) ndarray of pixels in images 2 that're matched to pixels1
    :param projMatr1: projection matrix from WORLD to image 1, (3,4) ndarray
    :param projMatr2: projection matrix from WORLD to image 2, (3,4) ndarray
    :return: pc_3d: a (3,n) ndarray of the point cloud - landmarks in WORLD coordinate-system
    """
    assert pixels1.shape == pixels2.shape
    assert projMatr1.shape == projMatr2.shape == (3, 4)
    num_points = pixels1.shape[1]
    pc_4d = np.zeros((4,num_points)) # point cloud in homogeneous coordinates
    p1,p2,p3 = projMatr1
    p1_, p2_, p3_ = projMatr2
    for i in range(num_points):
        x,y = pixels1[:, i]
        x_,y_ = pixels2[:, i]
        A = np.vstack((p3*x - p1, p3*y - p2, p3_*x_- p1_, p3_*y_ - p2_ ))
        u,s,vh = np.linalg.svd(A)
        X = vh[-1,:] # (4,)

        # iterative Least Squares
        for j in range(50):
            first_eq = X @ p3
            second_eq = X @ p3_
            B = np.vstack((A[:2]*first_eq, A[2:]*second_eq))
            u, s, vh = np.linalg.svd(B)
            X = vh[-1, :]  # (4,)

        pc_4d[:,i] = X
    pc_3d = pc_4d[:-1,:] / (pc_4d[-1].reshape(1,-1)) # (3,n) ndarray
    return pc_3d


#########################   Visualization Tools  #########################
def draw_point_cloud(pc, title="", save=False, inliers_bool=None, hist=False):
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
        path = sys_utils.get_avail_path(os.path.join(sys_utils.out_dir(), f'{title}_pc.png'))
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

def draw_triangulation(img, pc, pxls, title="", save=False):
    """ :param pc: (3,num_of_matches)
        :param pxls: (2,num_of_matches) """
    assert pc.shape[1] == pxls.shape[1]
    draw_point_cloud(pc=pc, title=title, save=save)
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
    utils.image.cv_show_img(img, title=f"{title}_vis_tr", save=save)
