import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import utils

def get_quantile_point_cloud_filter(pc):
    pc_abs = np.abs(pc)
    x_quant, y_quant, z_quant = np.quantile(pc_abs, q=0.99, axis=1)
    x_crit = (pc_abs[0] <= x_quant)
    y_crit = (pc_abs[1] <= y_quant)
    z_crit = (pc_abs[2] <= z_quant)
    filtered = x_crit * y_crit * z_crit
    return filtered

def get_relative_point_cloud_filter(pc):
    """ :param pc: (3,n) """
    x_abs = np.abs(pc[0]); y_abs = np.abs(pc[1])
    x_crit = (x_abs <= 50)
    y_crit = (y_abs <= 50)
    z_crit1 = pc[2] < 200
    z_crit2 = pc[2] > 1
    z_crit = z_crit1 * z_crit2
    filtered = (x_crit * y_crit) * z_crit
    return filtered

def filter_based_on_triang(kp_l, desc_l, kp_r, pc):
    filtered = get_relative_point_cloud_filter(pc)
    return kp_l[:,filtered], desc_l[filtered], kp_r[:, filtered], pc[:,filtered]

def triang(kpA, kpB, k, mA, mB):
    """
    get 3D (World) coordinates via triangulation of matched pixels from two images
    :param kpA/B: pixels of matched points in imageA/B, (2,n) array
    :param k: intrinsics  matrix shared by camera A/B, (3,4)
    :param mA/B: extrinsics matrices of camera A/B (4,4)
    :return: (3,n) ndarray of world coordinates in coordinates frame of cameraA
    """
    assert kpA.shape == kpB.shape
    pA = k @ mA  # (3,4) # projection matrix
    pB = k @ mB  # (3,4)

    pc_4d = cv2.triangulatePoints(projMatr1=pA, projMatr2=pB, projPoints1=kpA, projPoints2=kpB)  # (4,n)
    pc_3d = pc_4d[0:3] / pc_4d[-1]  # (3,n)
    return pc_3d

def triang_from_keypoints(kp0, kp1, k, m0, m1):
    """ :param kp0/1: [KeyPoint1, .... , KeypointN] """
    assert len(kp0) == len(kp1)
    pxls0 = np.array([kp.pt for kp in kp0]).T  # (2,n_matches)
    pxls1 = np.array([kp.pt for kp in kp1]).T  # (2,n_matches)
    pc_3d = triang(kp0=pxls0, kp1=pxls1, k=k, m0=m0, m1=m1)  # (3,n_matches)
    return pc_3d

#########################   Visualization utils  #########################
def vis_pc(pc, title="", save=False,cameras=None):
    """ :param pc: (3,num_of_matches), inhomogeneous"""
    assert pc.shape[0] in [3,4]
    if pc.shape[0] == 4:
        pc = pc[:3] / pc[-1,:]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pc[0, :], pc[2, :], pc[1, :], color="blue") # this isn't a mistake, plt's z axis is our Y axis
    if cameras is not None:
        ax.scatter(cameras[0,:],cameras[2,:], cameras[1,:],marker=(5,2), color="red", s=50)
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
        path = utils.get_avail_path(os.path.join(utils.fig_path(), f'{title}_pc.png'))
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

def vis_triang(img, pc, pxls, title="", save=False):
    """ :param pc: (3,num_of_matches), inhomogeneous
        :param pxls: (2,num_of_matches) inhomogeneous """
    assert pc.shape[1] == pxls.shape[1]
    vis_pc(pc=pc,title=title, save=save)
    # plot pxls with txt indicating their 3d location
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    num_matches = pc.shape[1]
    rand_inds = np.random.randint(0, num_matches, size=10)
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
    import kitti
    import features
    pc_3d = np.vstack((np.arange(11),np.arange(11),np.arange(11)))
    vis_pc(pc=pc_3d, save=False)
    # fig0 = plt.figure(); plt.plot(np.arange(10),np.arange(10))
    fig1 = plt.figure(1); axes1 = fig1.gca()
    axes1.scatter(np.array(50),np.array(50),np.array(50))
    plt.show()

