import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import my_code.utils as my_utils

def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(os.path.join(my_utils.DATA_PATH, 'image_0', img_name), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(my_utils.DATA_PATH, 'image_1', img_name), cv2.IMREAD_GRAYSCALE)
    return img1, img2

def plt_disp_img(img,name, save=False):
    plt.axis('off'); plt.margins(0, 0)
    plt.title(name)
    plt.imshow(img)
    if save:
        plt.savefig(os.path.join(my_utils.FIG_PATH, name + '.png'), bbox_inches='tight', pad_inches=0)
    plt.show()

def cv_disp_img(img, title='', save=False):
    cv2.imshow(title, img); cv2.waitKey(0); cv2.destroyAllWindows()
    if save:
        title = title if title else 'res'
        cv2.imwrite(f'{title}.png', img)

def draw_kp(img, to_plot=True):
    orb = cv2.ORB_create(nfeatures=1000)
    kp, desc = orb.detectAndCompute(img1, None)
    if to_plot:
        kp_img = cv2.drawKeypoints(img, kp,None, color=(0, 255, 0), flags=0)
        plt_disp_img(img=kp_img, name=f'{orb.getDefaultName()}_keypoints', save=True)
    for i in range(2):
        print(f'---- desc {i} -----')
        print(desc[i])
    return kp,desc

MATCH_Y_DIST_MAX = 2

def match_images(img1, img2, sig=True, to_plot=True):
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, desc1 = orb.detectAndCompute(img1, None) # query
    kp2, desc2 = orb.detectAndCompute(img2, None) # train
    img_name = f'{orb.getDefaultName()}'
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=sig)
    matches = bf.match(desc1, desc2)

    # filter matches based on y distance
    filtered_matches = []
    pts1, pts2 = [], []
    for match in matches:
        y_dist = abs(kp1[match.queryIdx].pt[1] - kp2[match.trainIdx].pt[1])
        if y_dist <= MATCH_Y_DIST_MAX:
            pts1.append(kp1[match.queryIdx].pt)
            pts2.append(kp2[match.trainIdx].pt)
            filtered_matches.append(match)
    # filtered_matches = sorted(filtered_matches, key=lambda x: x.distance)
    pts1 = np.array(pts1).T
    pts2 = np.array(pts2).T
    # plt.hist(dists, bins="stone");plt.title(f"histogram of matches by y dist, {sum(dists>2)/len(dists):.1%} larger than 2 px diff")
    # plt.ylabel("number of matches"); plt.xlabel("match's distance in y-axis"); plt.show()
    # print(f"percentage of matches larger than 2 px diff: {sum(dists>2)/len(dists):.1%}")

    # plot incorrect
    # color = CYAN_COLOR
    # img_name += '_incorrect'
    # incorrect_mathces = []
    # for match in reversed(matches):
    #     img1_kp = kp1[match.queryIdx]
    #     img2_kp = kp2[match.trainIdx]
    #     col1, row1 = img1_kp.pt
    #     col2, row2 = img2_kp.pt
    #     if abs(row2 - row1) > 30:
    #         incorrect_mathces.append(match)
    #     if len(incorrect_mathces)>50:
    #         break
    if to_plot:
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
        plt_disp_img(img=match_img, name=img_name, save=True)
    return filtered_matches, pts1, pts2

def read_cameras():
    # read camera matrices
    with open(os.path.join(my_utils.DATA_PATH,'calib.txt')) as f:
        l1 = f.readline().split()[1:]
        l2 = f.readline().split()[1:]
    p1 = np.array([float(i) for i in l1]).reshape(3,4)
    p2 = np.array([float(i) for i in l2]).reshape(3,4)
    k = p1[:, :3]
    m1 = np.linalg.inv(k) @ p1
    m2 = np.linalg.inv(k) @ p2
    return k, m1,m2


def triangulate(pts1, pts2, m1, m2):
    assert pts1.shape == pts2.shape
    assert m1.shape == m2.shape == (3,4)
    num_points = pts1.shape[1]
    new_points = np.zeros((4,len(matches)))
    p1,p2,p3 = m1
    p1_, p2_, p3_ = m2
    for i in range(num_points):
        x,y = pts1[:,i]
        x_,y_ = pts2[:, i]
        r1 = p3*x - p1
        r2 = p3*y - p2
        r3 = p3_*x_- p1_
        r4 = p3_*y_ - p2_
        # A = np.vstack((p3*x - p1, p3*y - p2, p3_*x_- p1_, p3_*y_ - p2_ ))
        A = np.array([r1,r2,r3,r4])
        u,s,vh = np.linalg.svd(A)
        X = vh[-1,:] # (4,)
        for j in range(50):
            first_eq = X @ p3
            second_eq = X @ p3_
            B = np.vstack((A[:2]*first_eq, A[2:]*second_eq))
            u, s, vh = np.linalg.svd(B)
            X = vh[-1, :]  # (4,)
        new_points[:,i] = X
    inhom_points = new_points[:-1,:] / (new_points[-1].reshape(1,-1)) # (3,n)
    return inhom_points

def vis_triangulation(img1, cv2_3d, pts1, title=""):
    """ :param cv2_3d: (3,num_of_matches), inhomogeneous
        :param pts1: (2,num_of_matches) inhomogeneous """
    assert cv2_3d.shape[1] == pts1.shape[1]
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    num_matches = cv2_3d.shape[1]
    rand_inds = np.random.randint(0, num_matches, size=10)
    for ind in rand_inds:
        x_w, y_w, z_w = cv2_3d[0:3, ind]
        x, y = pts1[:, ind]
        x = int(x); y = int(y)
        print(f'({x},{y:}) -> ({x_w:.1f},{y_w:.1f},{z_w:.1f})')
        img1 = cv2.circle(img1, (x, y), 1, (0, 0, 0))
        img1 = cv2.putText(img1, f'{x_w:.1f},{y_w:.1f},{z_w:.1f}', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                           color=(0, 0, 255), lineType=cv2.LINE_AA)
    cv_disp_img(img1, title=f"vis_triang_{title}",save=True)

def vis_triangulation_compare(img1, cv2_3d, my_3d, pts1, title=""):
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
        cv_img1 = cv2.circle(cv_img1, (x, y), 1, (0, 0, 0))
        cv_img1 = cv2.putText(cv_img1, f'{cv_x_w:.1f},{cv_y_w:.1f},{cv_z_w:.1f}', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 255))
        my_img1 = cv2.circle(my_img1, (x, y), 1, (0, 0, 0))
        my_img1 = cv2.putText(my_img1, f'{my_x_w:.1f},{my_y_w:.1f},{my_z_w:.1f}', (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=(0, 0, 255))
    cv2.imwrite("vis_traing_my_com.png", my_img1)
    cv2.imwrite("vis_traing_cv2_com.png", cv_img1)
    # cv_disp_img(img1, title=f"vis_triang_{title}",save=True)

if __name__=="__main__":
    img1, img2 = read_images(idx=0)
    # Q1.1- Q1.2
    # kp1 = draw_kp(img=img1, to_plot=True)
    # kp2 = draw_kp(img=img2, to_plot=True)


    matches, pts1, pts2 = match_images(img1, img2, sig=True, to_plot=False)  # list(n), (2Xn), (2xn)
    k, m1, m2 = read_cameras()  # k=(3,3) m1,m2 (3,4)
    p1 = k@m1
    p2 = k@m2
    cv2_4d = cv2.triangulatePoints(projMatr1=p1, projMatr2=p2, projPoints1=pts1, projPoints2=pts2) # (4,n)
    cv2_3d = cv2_4d[:-1] / (cv2_4d[-1].reshape(1, -1)) # (3,n)
    my_3d_points = triangulate(pts1, pts2, p1, p2)  # (3,n)
    vis_triangulation(img1, cv2_3d, pts1, title="cv2")
    # vis_triangulation(img1, my_3d_points, pts1, title="mine")
    # vis_triangulation_compare(img1, cv2_3d, my_3d_points, pts1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(my_3d_points[0,:],my_3d_points[1,:],my_3d_points[2,:])
    ax.set_title('mine')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.savefig(os.path.join(my_utils.FIG_PATH, 'triang_mine.png'), bbox_inches='tight', pad_inches=0)
    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title('cv2')
    ax.scatter(cv2_3d[0,:], cv2_3d[1,:], cv2_3d[2,:])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.savefig(os.path.join(my_utils.FIG_PATH, 'triang_cv22.png'), bbox_inches='tight', pad_inches=0)
    plt.show()

    print('end')



