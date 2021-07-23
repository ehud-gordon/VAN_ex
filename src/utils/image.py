import cv2
from matplotlib import pyplot as plt

import os
import utils.sys_utils as sys_utils

def plt_show_img(img, name="", plot_dir="", save=False):
    plt.axis('off'); plt.margins(0, 0)
    plt.title(name)
    plt.imshow(img, cmap='gray')
    if save:
        name = name if name else 'img'
        path = os.path.join(plot_dir, name + '.png')
        path = sys_utils.get_avail_path(path)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()

def cv_show_img(img, title='', plot_dir="", save=False):
    cv2.imshow(title, img); cv2.waitKey(0); cv2.destroyAllWindows()
    if save:
        title = title if title else 'res'
        path = os.path.join(plot_dir, title +'.png')
        path = sys_utils.get_avail_path(path)
        cv2.imwrite(path, img)

def bgr_rgb(img):
    """ swithces between bgr and rgb"""
    if img.ndim != 3:
        print("error rgb_bgr")
        return img
    return img[:, :, ::-1]

def image_to_grid(img, num_rows=2, num_cols=2):
    """
    |--------- <x=1226>
    |
    |
    |
    <y=370>
    :return:  a list of tuples: [(cell, origin) ,...,()], where origin is tuple(r,c) of the
    upper left corner of the cell, and cell is an (n,m) ndarray, containing  the pixels of the image for this cell.
    """
    grid = []
    row_step= img.shape[0]//num_rows # along the vertical axis ## # YYY = ROW
    col_step = img.shape[1]//num_cols #### XXXX == COL
    for r in range(num_rows):
        for c in range(num_cols):
            origin = [row_step * r, col_step * c]  # [r,c]
            cell = img[row_step*r:row_step*(r+1), col_step*c:col_step*(c+1)]
            grid.append((cell,origin))
    return grid