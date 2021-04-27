import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pathlib

DATA_PATH = r'C:\Users\godin\Documents\VAN_ex\data\dataset05\sequences\05'
FIG_PATH = r'C:\Users\godin\Documents\VAN_ex\fig'
CYAN_COLOR = (255,255,0) # in BGR
ORANGE_COLOR = (0, 128,255) # in BGR
MATCH_Y_DIST_MAX = 2

def dir_name_ext(path):
    dir, base = os.path.split(path)
    name, ext = os.path.splitext(base)
    return dir, name, ext

def get_avail_path(path):
    while os.path.exists(path):
        dir,name,ext = dir_name_ext(path)
        path = os.path.join(dir, name+'0'+ext)
    return path


def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    # img1 = cv2.imread(os.path.join(DATA_PATH, 'image_0', img_name)) # left image
    # img2 = cv2.imread(os.path.join(DATA_PATH, 'image_1', img_name)) # right image
    img1 = cv2.imread(os.path.join(DATA_PATH, 'image_0', img_name), cv2.IMREAD_GRAYSCALE)  # left image
    img2 = cv2.imread(os.path.join(DATA_PATH, 'image_1', img_name), cv2.IMREAD_GRAYSCALE)  # right image
    return img1, img2

def read_cameras():
    """
    :return: k - intrinsic matrix (shared by both cameras), np.array 3x3
             m1 - extrinstic camera matrix [R|t] of left camera, np.array 3x4
             m2 - extrinstic camera matrix [R|t] of right camera, np.array 3x4
             to get camera matrix (P), compute k @ m1
    """
    # read camera matrices
    with open(os.path.join(DATA_PATH,'calib.txt')) as f:
        l1 = f.readline().split()[1:]
        l2 = f.readline().split()[1:]
    p1 = np.array([float(i) for i in l1]).reshape(3,4)
    p2 = np.array([float(i) for i in l2]).reshape(3,4)
    k = p1[:, :3]
    m1 = np.linalg.inv(k) @ p1
    m2 = np.linalg.inv(k) @ p2
    return k, m1,m2


##################       Visualization     #####################
def plt_disp_img(img, name, save=False):
    plt.axis('off'); plt.margins(0, 0)
    plt.title(name)
    plt.imshow(img)
    if save:
        path = os.path.join(FIG_PATH, name + '.png')
        path = get_avail_path(path)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

def cv_disp_img(img, title='', save=False):
    cv2.imshow(title, img); cv2.waitKey(0); cv2.destroyAllWindows()
    if save:
        title = title if title else 'res'
        path = os.path.join(FIG_PATH, title +'.png')
        path = get_avail_path(path)
        cv2.imwrite(path, img)

def draw_matches(img1, kp1, img2, kp2, matches):
    matches_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv_disp_img(img=matches_img, title="matches_20", save=True)

def vis_matches(img1, img2, kp1, kp2, matches):
    # get first 20 matches
    if isinstance(matches[0], list):
        matches = [match[0] for match in matches]
    first_matches = matches[0:20]
    draw_matches(img1=img1, kp1=kp1, img2=img2, kp2=kp2, matches=first_matches)
    y_dists = []
    for match in matches:
        y_dist = abs(kp1[match.queryIdx].pt[1] - kp2[match.trainIdx].pt[1])
        y_dists.append(y_dist)

    # plot histogram
    y_dists = np.array(y_dists)
    plt.hist(y_dists, bins="stone")
    plt.title(f"hist of y-dist of matches,"
              f" {sum(y_dists>2)}/{len(matches)}={sum(y_dists>2)/len(matches):.1%} > 2")
    plt.ylabel("number of matches"); plt.xlabel("match's distance in y-axis")
    hist_path = get_avail_path(os.path.join(FIG_PATH, "matches_hist.png"))
    plt.savefig(hist_path)
    plt.show()
    print(f"{sum(y_dists>2)}/{len(matches)}={sum(y_dists>2)/len(matches):.1%} matches with y-dist > 2:")

def plot_inliers_outliers(all_matches, inliers, kp1, kp2, img1, img2):
    outliers = [match for match in all_matches if match not in inliers]

    inliers_kp1 = [kp1[match.queryIdx] for match in inliers]
    outliers_kp1 = [kp1[match.queryIdx] for match in outliers]

    inliers_kp2 = [kp2[match.trainIdx] for match in inliers]
    outliers_kp2 = [kp2[match.trainIdx] for match in outliers]

    img1 = cv2.drawKeypoints(image=img1, keypoints=inliers_kp1,outImage=None, color=ORANGE_COLOR, flags=0)
    img1 = cv2.drawKeypoints(image=img1, keypoints=outliers_kp1, outImage=None, color=CYAN_COLOR, flags=0)
    cv_disp_img(img=img1, title="left image inliers(orange) and outliers(cyan)", save=True)

    img2 = cv2.drawKeypoints(image=img2, keypoints=inliers_kp2, outImage=None, color=ORANGE_COLOR, flags=0)
    img2 = cv2.drawKeypoints(image=img2, keypoints=outliers_kp2, outImage=None, color=CYAN_COLOR, flags=0)
    cv_disp_img(img=img2, title="right image inliers(orange) and outliers(cyan)", save=True)

def filter_cloud_points(points_3d, pts1):
    assert points_3d.shape[1] == pts1.shape[1]
    quantile = np.quantile(points_3d[2, :], q=0.98)
    filter = np.logical_and(points_3d[2, :] > 1, points_3d[2, :] < quantile)
    points_3d = points_3d[:, filter]
    pts1 = pts1[:, filter]

    return points_3d, pts1

def vis_triangulation(img1, points_3d, pts1, title=""):
    """ :param points_3d: (3,num_of_matches), inhomogeneous
        :param pts1: (2,num_of_matches) inhomogeneous """
    assert points_3d.shape[1] == pts1.shape[1]
    points_3d, pts1 = filter_cloud_points(points_3d=points_3d, pts1=pts1)

    # cloud points
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points_3d[0, :], points_3d[2, :], points_3d[1, :]) # this isn't a mistake, plt's z axis is our Y axis
    ax.set_title(title)
    xmin, ymin, zmin = np.min(points_3d, axis=1)
    xmax, ymax, zmax = np.max(points_3d, axis=1)
    ax.set_ylim([0, zmax + 1])  # not a mistake, plt's Y axis is our Z-Axis
    ax.set_xlim([xmin - 1, xmax + 1])
    ax.set_zlim([ymin - 1, ymax + 1]) # not a mistake, plt's z-axis is our Y-axis
    ax.invert_zaxis()  # not a mistake, - plt's z axis is our Y axis
    ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y') # not a mistake
    path = get_avail_path(os.path.join(FIG_PATH, f'cloud_point_{title}.png'))
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    num_matches = points_3d.shape[1]
    rand_inds = np.random.randint(0, num_matches, size=10)
    for ind in rand_inds:
        x_w, y_w, z_w = points_3d[0:3, ind]
        x, y = pts1[:, ind]
        x = int(x); y = int(y)
        print(f'({x},{y:}) -> ({x_w:.1f},{y_w:.1f},{z_w:.1f})')
        img1 = cv2.circle(img1, (x, y), 2, (0, 0, 0))
        img1 = cv2.putText(img1, f'{x_w:.1f},{y_w:.1f},{z_w:.1f}', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                           color=(0, 0, 255), lineType=cv2.LINE_AA)
    print()

    cv_disp_img(img1, title=f"vis_triangulation_{title}",save=True)

def vis_triangulation_compare(img1, cv2_3d, my_3d, pts1):
    """ :param cv2_3d: (3,num_of_matches), inhomogeneous
        :param pts1: (2,num_of_matches) inhomogeneous """
    assert cv2_3d.shape[1] == pts1.shape[1]
    cv_img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    my_img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    num_matches = cv2_3d.shape[1]
    rand_inds = np.random.randint(0, num_matches, size=5)
    for ind in rand_inds:
        x, y = pts1[:, ind]
        x = int(x); y = int(y)
        cv_x_w, cv_y_w, cv_z_w = cv2_3d[:, ind]
        my_x_w, my_y_w, my_z_w = my_3d[:,ind]

        print(f'({x},{y:}) -> cv:({cv_x_w:.1f},{cv_y_w:.1f},{cv_z_w:.1f}), my:({my_x_w:.1f},{my_y_w:.1f},{my_z_w:.1f})')
        cv_img1 = cv2.circle(cv_img1, (x, y), 2, (0, 0, 0))
        cv_img1 = cv2.putText(cv_img1, f'{cv_x_w:.1f},{cv_y_w:.1f},{cv_z_w:.1f}', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 255))
        my_img1 = cv2.circle(my_img1, (x, y), 2, (0, 0, 0))
        my_img1 = cv2.putText(my_img1, f'{my_x_w:.1f},{my_y_w:.1f},{my_z_w:.1f}', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 255))
    cv2.imwrite("1vis_traing_my_com.png", my_img1)
    cv2.imwrite("1vis_traing_cv2_com.png", cv_img1)
    # cv_disp_img(img1, title=f"vis_triang_{title}",save=True)