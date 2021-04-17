import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from my_code.utils import *

def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(os.path.join(DATA_PATH, 'image_0', img_name), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(DATA_PATH, 'image_1', img_name), cv2.IMREAD_GRAYSCALE)
    return img1, img2

def plt_disp_img(img,name, save=False):
    plt.axis('off'); plt.margins(0, 0)
    plt.title(name)
    plt.imshow(img)
    if save:
        plt.savefig(os.path.join(FIG_PATH, name + '.png'), bbox_inches='tight', pad_inches=0)
    plt.show()

def cv_disp_img(img, title=''):
    cv2.imshow(title, img); cv2.waitKey(0); cv2.destroyAllWindows()

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

def match_images(img1, img2, sig=True, to_plot=True):
    orb = cv2.ORB_create()
    kp1, desc1 = orb.detectAndCompute(img1, None) # query
    kp2, desc2 = orb.detectAndCompute(img2, None) # train
    img_name = f'{orb.getDefaultName()}'
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=sig)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.zeros((2,len(matches)))
    pts2 = np.zeros((2,len(matches)))
    for i, match in enumerate(matches):
        pts1[:,i] =  kp1[match.queryIdx].pt
        pts2[:, i] = kp2[match.trainIdx].pt
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
    return matches, pts1, pts2

def read_cameras():
    # read camera matrices
    with open(os.path.join(DATA_PATH,'calib.txt')) as f:
        l1 = f.readline().split()[1:]
        l2 = f.readline().split()[1:]
    m1 = np.array([float(i) for i in l1]).reshape(3,4)
    m2 = np.array([float(i) for i in l2]).reshape(3,4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1,m2


def triangulate(pts1, pts2, m1, m2):
    num_points = pts1.shape[1]
    new_points = np.zeros((len(matches), 4))
    p1,p2,p3 = m1
    p1_, p2_, p3_ = m2
    for i in range(num_points):
        x,y = pts1[:,i]
        x_,y_ = pts2[:, i]
        A = np.vstack((
            [p3*x - p1],
            [p3*y - p2],
            [p3_*x_- p1_],
            [p3_*y_ - p2_]
        ))
        u,s,vh = np.linalg.svd(A)
        X = vh[2,:] # (4,)
        for j in range(50):
            first_eq = X @ p3
            second_eq = X @ p3_
            A = np.vstack((A[:2]*first_eq, A[2:]*second_eq))
            u, s, vh = np.linalg.svd(A)
            X = vh[2, :]  # (4,)
        new_points[i] = X.T
    inhom_points = new_points[:,:-1] / (new_points[:,-1].reshape(-1,1)) # (n,3)
    return inhom_points


if __name__=="__main__":
    img1, img2 = read_images(idx=0)
    # Q1.1- Q1.2
    # kp1 = draw_kp(img=img1, to_plot=True)
    # kp2 = draw_kp(img=img2, to_plot=True)
    matches, pts1, pts2 = match_images(img1, img2, sig=True, to_plot=False) # list(n), (2Xn), (2xn)
    k, m1, m2 = read_cameras() #k=(3,3) m1,m2 (3,4)
    inhom_points = triangulate(pts1,pts2, m1, m2) # (n,3)
    cv2_points = cv2.triangulatePoints(projMatr1=m1, projMatr2=m2, projPoints1=pts1, projPoints2=pts2) # (4,n)
    cv2_inhom = cv2_points[:-1] / (cv2_points[-1].reshape(1, -1)) # (3,n)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(inhom_points[:,0],inhom_points[:,1],inhom_points[:,2])
    ax.set_title('mine')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title('cv2')
    ax.scatter(cv2_inhom[0,:], cv2_inhom[1,:], cv2_inhom[2,:])
    plt.show()
    print('end')



