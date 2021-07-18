"""
Visualization utils for features2d (keypoints) and matching
"""

import cv2
import numpy as np
import scipy.spatial
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch

import os

class DrawMatchesDouble:
    def __init__(self, img0, img1, kp0, kp1, pc=None):
        if img0.ndim == 2:
            img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        self.img0 = img0
        self.img1 = img1
        self.kp0 = kp0
        self.kp1 = kp1
        self.ckdtree0 = scipy.spatial.cKDTree(kp0.T)
        self.ckdtree1 = scipy.spatial.cKDTree(kp1.T)
        self.con_ind = 0
        self.cons = []
        self.curr_cons = []
        self.pc=pc

    def on_hover(self, event):
        if event.inaxes is None:
            return
        if event.inaxes == self.ax0:
            print(event.x)
            print(event.y)
        else:
            pass

    def clear_cons(self):
        [conp.remove() for conp in self.curr_cons]
        self.curr_cons = []

    def onkeypress(self,event):
        if event.key == 'c': # clear all
            self.clear_cons()
        if event.key == 'd': # draw all
            if len(self.curr_cons) == len(self.cons):
                return
            else:
                self.clear_cons()
                [self.ax1.add_artist(con) for con in self.cons]
                self.curr_cons = self.cons
        if event.key == 'n': # draw only next
            self.clear_cons()
            self.curr_cons = [self.cons[self.con_ind]]
            self.ax1.add_artist(self.cons[self.con_ind])
            self.con_ind  = (self.con_ind + 1) % len(self.cons)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def onmouseclick(self,event):
        if event.inaxes is None:
            return
        if event.inaxes == self.ax1:
            closest_index = self.ckdtree0.query([event.xdata,event.ydata])[1]
        else:
            closest_index = self.ckdtree1.query([event.xdata, event.ydata])[1]
        xy0 = self.kp0[:,closest_index]
        xy1 = self.kp1[:, closest_index]
        self.clear_cons()
        if self.pc is not None: print(f'{self.pc[:,closest_index]}')
        con = ConnectionPatch(xyA=tuple(xy0), xyB=tuple(xy1), coordsA="data", coordsB="data",
                                  axesA=self.ax0, axesB=self.ax1, alpha=0.5, color="blue")
        self.curr_cons = [con]
        self.ax1.add_artist(con)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw_matches_double(self,size=0, title="", save=False, plot_dir=""):
        self.fig, (self.ax0, self.ax1) = plt.subplots(1, 2)
        self.fig.canvas.mpl_connect('button_press_event', self.onmouseclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        plt.suptitle(f"matches_{title}")
        self.ax0.imshow(self.img0)
        self.ax1.imshow(self.img1)
        inds = range(self.kp1.shape[1])
        if size:
            inds = np.random.choice(self.kp0.shape[1], size=size, replace=False)
        for i in inds:
            xy0 = self.kp0[:, i]
            xy1 = self.kp1[:, i]
            con = ConnectionPatch(xyA=tuple(xy0), xyB=tuple(xy1), coordsA="data", coordsB="data",
                                  axesA=self.ax0, axesB=self.ax1, alpha=1, color="blue")
            self.cons.append(con)
            self.ax1.add_artist(con)
            self.ax0.plot(xy0[0], xy0[1], f'ro', markersize=3)
            self.ax1.plot(xy1[0], xy1[1], f'ro', markersize=3)
        self.curr_cons = self.cons
        self.fig.subplots_adjust(left=0.03, bottom=0.19, right=0.99, top=0.94, wspace=0.04, hspace=0.2)
        if save:
            path = os.path.join(plot_dir, f'matches_{title}' + '.png')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.show()