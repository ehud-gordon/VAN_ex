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
    if sig:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # img_name += '_sig'
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # img_name += '_no_sig'
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
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
        img_name+='_correct'
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:30], None,matchColor=ORANGE_COLOR, flags=2)
        plt_disp_img(img=match_img, name=img_name, save=True)

    return matches


if __name__=="__main__":
    img1, img2 = read_images(idx=0)
    # Q1.1- Q1.2
    # kp1 = draw_kp(img=img1, to_plot=True)
    # kp2 = draw_kp(img=img2, to_plot=True)
    match_images(img1, img2, sig=True)



