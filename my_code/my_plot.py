import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import plot as plotly_plot
import plotly.graph_objects as go
import os

from utils import und_title

############ MATPLOTLIB ############

def plt_diff_trans_vecs(trans_diff, plot_dir,title, idx=None, plot=False, save=True):
    """ :param trans_diff: (3,n)  """
    idx = np.arange(1,trans_diff.shape[1]+1) if idx is None else idx
    fig = plt.figure()
    plt.plot(idx, trans_diff[0], label="tx")
    plt.plot(idx, trans_diff[1], label="ty")
    plt.plot(idx, trans_diff[2], label="tz")
    plt.ylabel("meters"); plt.xlabel("frames")
    plt.title (f"Relative abs diff btwn my and kitti trans vectors, {title} in [{idx[0]}-{idx[-1]}]")
    plt.legend()
    if save:
        path = os.path.join(plot_dir, f"diff_trans{und_title(title)}{idx[0]}_{idx[-1]}.png")
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()

def plt_diff2_trans_vecs(my_vecs, kitti_vecs, plot_dir, title ,idx=None, plot=False, save=True):
    assert my_vecs.shape[1] == kitti_vecs.shape[1]
    idx = np.arange(1, my_vecs.shape[1] + 1) if idx is None else idx
    diff = np.abs(my_vecs - kitti_vecs)
    fig = plt.figure()
    plt.plot(idx, diff[0], label="tx")
    plt.plot(idx, diff[1], label="ty")
    plt.plot(idx, diff[2], label="tz")
    plt.ylabel("meters"); plt.xlabel("frames")
    plt.title (f"Relative abs diff btwn my and kitti trans vectors, {title} in [{idx[0]}-{idx[-1]}]")
    plt.legend()
    if save:
        path = os.path.join(plot_dir, f"diff_trans{und_title(title)}{idx[0]}_{idx[-1]}.png")
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()

def plt_diff_rot_matrices(rot_diff, plot_dir,title, idx=None, plot=False, save=True):
    """ :param rot_diff:   """
    idx = np.arange(1,len(rot_diff)+1) if idx is None else idx #[1,10]
    fig = plt.figure()
    plt.plot(idx, rot_diff)
    plt.ylabel("degrees"); plt.xlabel("frames")
    plt.title (f"Relative diff btwn my and kitti rotation matrices, {title} in [{idx[0]}-{idx[-1]}]")
    if save:
        path = os.path.join(plot_dir, f"diff_rot{und_title(title)}{idx[0]}_{idx[-1]}.png")
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()

def plt_2d_cams(camera_dws, plot_dir,title, startframe=0, endframe=0, plot=False, save=True):
    endframe = camera_dws.shape[1]-1 if not endframe else endframe # 10
    plt.figure()
    plt.scatter(x=camera_dws[0], y=camera_dws[2], marker=(5,2), color="red")
    plt.xlabel('x');plt.ylabel('z')
    plt.title(f"2D camera's locations {title} in [{startframe}-{endframe}]")
    if save:
        path = os.path.join(plot_dir, f"2d_cams{und_title(title)}{startframe}_{endframe}" + '.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()

def plt_2d_cams_compare(my_dws, kitti_dws, plot_dir, title,endframe=0, startframe=0, plot=False, save=True):
    endframe = my_dws.shape[1]-1 if not endframe else endframe # 10
    plt.figure()
    plt.scatter(x=my_dws[0], y=my_dws[2], color="blue", label="mine")
    plt.scatter(x=kitti_dws[0], y=kitti_dws[2], color="red", label="kitti", alpha=0.4)
    plt.xlabel('X');plt.ylabel('Z')
    plt.title(f"2D camera's locations comparison, {title} in frames [{startframe}-{endframe}]")
    plt.legend()
    if save:
        path = os.path.join(plot_dir, f'2d_cams_comp{und_title(title)}{startframe}_{endframe}' + '.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()

def plt_3d_cams(camera_dws, plot_dir, title, startframe=0, endframe=0, plot=False, save=True):
    endframe = camera_dws.shape[1]-1 if not endframe else endframe # 10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(camera_dws[0], camera_dws[2], camera_dws[1], color="blue")
    xmin, ymin, zmin = np.min(camera_dws, axis=1)
    xmax, ymax, zmax = np.max(camera_dws, axis=1)
    ax.set_ylim([0, zmax + 1])  # not a mistake, plt's Y axis is our Z-Axis
    ax.set_xlim([xmin - 1, xmax + 1])
    ax.set_zlim([ymin - 1, ymax + 1])  # not a mistake, plt's z-axis is our Y-axis
    ax.invert_zaxis()  # not a mistake, - plt's z axis is our Y axis
    ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y')  # not a mistake
    plt.title(f"left camera location, {title} in frames [{startframe}-{endframe}]")
    if save:
        path = os.path.join(plot_dir, f'3d_cams{und_title(title)}_{endframe}' + '.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()

def plt_3d_cams_compare(my_dws, kitti_dws, plot_dir, title, endframe=0, startframe=0, plot=False, save=True):
    endframe = my_dws.shape[1]-1 if not endframe else endframe
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(my_dws[0], my_dws[2], my_dws[1], color="blue", label="mine")
    ax.scatter(kitti_dws[0], kitti_dws[2], kitti_dws[1], color="red", label="kitti", alpha=0.4)
    xmin, ymin, zmin = np.min(my_dws, axis=1)
    xmax, ymax, zmax = np.max(my_dws, axis=1)
    ax.set_ylim([0, zmax + 1])  # not a mistake, plt's Y axis is our Z-Axis
    ax.set_xlim([xmin - 1, xmax + 1])
    ax.set_zlim([ymin - 1, ymax + 1])  # not a mistake, plt's z-axis is our Y-axis
    ax.invert_zaxis()  # not a mistake, - plt's z axis is our Y axis
    ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y')  # not a mistake
    plt.legend()
    plt.title(f"Comparing left camera location, {title} in frames [{startframe}-{endframe}]")
    if save:
        path = os.path.join(plot_dir, f'3d_cams_comp{und_title(title)}{endframe}' + '.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()

def plt_bundle_errors(init_errors, final_errors, plot_dir, idx, plot=False, save=True):
    fig = plt.figure()
    plt.plot(idx, init_errors, label="before")
    plt.plot(idx, final_errors, label="after")
    plt.xlabel('frames')
    plt.ylabel('error')
    plt.title('Graph errors before and after optimization')
    plt.legend()
    if save:
        path = os.path.join(plot_dir, f'graph_errors_{idx[-1]}' + '.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()

def plt_kp_inlier_matches(kp_l1_inlier_matches,plot_dir, plot, save):
    endframe = len(kp_l1_inlier_matches)
    idx = np.arange(endframe)+1
    plt.figure()
    plt.plot(idx, [t[0] for t in kp_l1_inlier_matches], label="keypoints_left1")
    plt.plot(idx, [t[1] for t in kp_l1_inlier_matches], label="inliers")
    plt.plot(idx, [t[2] for t in kp_l1_inlier_matches], label="matches")
    plt.xlabel('frame'); plt.ylabel('count')
    plt.title('count of inliers / matches / (keypoints in left1)')
    plt.legend()
    if save:
        path = os.path.join(plot_dir, f'kp_l1_inlier_matches_{endframe}' + '.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()


############ PLOTLY ############
def plotly_bar(y, bins=None, title="", plot=True, plot_dir=None):
    if bins is None:
        bins = np.arange(len(y))
    fig = go.Figure(go.Bar(x=bins, y=y))
    fig.data[0].text = y
    fig.update_traces(textposition='inside', textfont_size=12)
    fig.update_layout(bargap=0, title_text=title, title_x=0.5, font=dict(size=18))
    fig.update_traces(marker_color='blue', marker_line_color='blue', marker_line_width=1)
    if plot_dir:
        path = os.path.join(plot_dir, f'bar_{und_title(title)}' + '.html')
        fig.write_html(path)
    if plot:
        plotly_plot(fig)
    
def plotly_hist(y, bins=None, title="", density=True, plot=True, plot_dir=None):
    if bins is None:
        bins = np.arange(np.max(y)+2)
    y_hist, bins = np.histogram(y, bins=bins, density=density)
    plotly_bar(y=y_hist, bins=bins, title=title, plot=plot, plot_dir=plot_dir)

def plotly_3d_cams_compare(my_dws, kitti_dws, plot_dir, title, endframe=0, save=True):
    endframe = my_dws.shape[1]-1 if not endframe else endframe
    fig = go.Figure()
    my_trace = go.Scatter3d(x=my_dws[0], y=my_dws[2], z=my_dws[1], name="mine", mode='markers', marker=dict(color="red"))
    kitti_trace = go.Scatter3d(x=kitti_dws[0], y=kitti_dws[2], z=kitti_dws[1], name="kitti", mode='markers', marker=dict(color="blue"))
    fig.add_traces([my_trace, kitti_trace])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), width=800, height=800)
    fig.update_layout(scene = dict(xaxis_title='X', yaxis_title='Z', zaxis_title='Y'))
    fig.update_layout(showlegend=True)
    fig.update_scenes(zaxis_autorange="reversed")
    title1 = f"Comparing left camera location, {title} in frames [0-{endframe}]"
    fig.update_layout(title_text=title1,  title_x=0.5, font=dict(size=14))

    if save:
        path = os.path.join(plot_dir, f'3d_cams_comp_plotly{und_title(title)}{endframe}' + '.html')
        fig.write_html(path)

def main():
    x = (0, 1, 3, 6, 8)
    y = (4, 5, 1, 8, 6)
    plotly_bar(y=y, bins=x, title="asd")

if __name__=="__main__":
    my_dws = np.array([[0, -0.004751, -0.01733, -0.02374, -0.02972, -0.03907, -0.05193, -0.06274, -0.07457, -0.09065,
                        -0.113, -0.1339, -0.1606, -0.1888, -0.2123, -0.2353, -0.2611, -0.2875, -0.308, -0.3268, -0.3465,
                        -0.3669, -0.3896, -0.4119, -0.4335, -0.4608, -0.4814, -0.4955, -0.517, -0.5386, -0.5611,
                        -0.5823, -0.6074, -0.6329, -0.6569, -0.6807, -0.7041, -0.7271, -0.7565, -0.7859, -0.8093,
                        -0.8333, -0.8557, -0.8819, -0.909, -0.9387, -0.9683, -0.9989, -1.031, -1.061, -1.093],[0, -0.01047, -0.02652, -0.03871, -0.05046, -0.06306, -0.08161, -0.09546, -0.1162, -0.1349,
                        -0.1567, -0.1749, -0.1893, -0.2049, -0.2205, -0.2383, -0.2554, -0.2636, -0.2664, -0.271,
                        -0.2803, -0.2863, -0.2884, -0.2893, -0.2887, -0.2915, -0.3015, -0.3129, -0.3239, -0.3437,
                        -0.3679, -0.3938, -0.4187, -0.445, -0.4759, -0.5089, -0.5426, -0.577, -0.6097, -0.6417, -0.6742,
                        -0.7039, -0.7336, -0.7621, -0.7896, -0.8157, -0.8407, -0.8659, -0.8908, -0.9154, -0.9394],[0, 0.5756, 1.154, 1.723, 2.291, 2.862, 3.444, 4.033, 4.628, 5.231, 5.842, 6.462, 7.104, 7.764,
                        8.442, 9.128, 9.819, 10.52, 11.24, 11.99, 12.76, 13.55, 14.36, 15.18, 16.01, 16.86, 17.72, 18.6,
                        19.5, 20.42, 21.35, 22.29, 23.24, 24.21, 25.19, 26.17, 27.17, 28.17, 29.19, 30.22, 31.25, 32.29,
                        33.34, 34.39, 35.46, 36.52, 37.6, 38.68, 39.77, 40.86, 41.95]])
    kitti_dws = np.array([[1.11e-16, 0.0035, 0.001174, -0.007832, -0.01508, -0.0204, -0.02969, -0.04153, -0.05594,
                           -0.07008, -0.08798, -0.1055, -0.1239, -0.1445, -0.1645, -0.1845, -0.2077, -0.233, -0.2538,
                           -0.2701, -0.2821, -0.2962, -0.3128, -0.3271, -0.3451, -0.3586, -0.3661, -0.3686, -0.3813,
                           -0.3931, -0.4066, -0.4197, -0.4376, -0.4507, -0.4686, -0.4871, -0.5056, -0.5259, -0.5465,
                           -0.5679, -0.5885, -0.6114, -0.6342, -0.6563, -0.68, -0.7046, -0.7318, -0.7509, -0.7637,
                           -0.7861, -0.812],[0, -0.009789, -0.02241, -0.04141, -0.05923, -0.07423, -0.09078, -0.1099, -0.1348, -0.1576,
                           -0.1803, -0.1997, -0.2188, -0.2366, -0.2543, -0.2738, -0.2924, -0.3036, -0.3092, -0.3152,
                           -0.3207, -0.3252, -0.3292, -0.33, -0.3299, -0.3311, -0.3374, -0.3507, -0.3643, -0.3836,
                           -0.4079, -0.4351, -0.4631, -0.4914, -0.5208, -0.5505, -0.5807, -0.6123, -0.6431, -0.6738,
                           -0.7048, -0.7345, -0.7659, -0.7961, -0.8251, -0.8521, -0.8792, -0.8999, -0.9174, -0.9426,
                           -0.9707], [2.22e-16, 0.5654, 1.128, 1.689, 2.251, 2.816, 3.39, 3.969, 4.556, 5.15, 5.751, 6.364, 6.99,
                           7.638, 8.302, 8.978, 9.662, 10.35, 11.06, 11.8, 12.56, 13.34, 14.12, 14.93, 15.75, 16.58,
                           17.43, 18.29, 19.17, 20.07, 20.99, 21.92, 22.88, 23.85, 24.82, 25.81, 26.79, 27.8, 28.8,
                           29.82, 30.84, 31.87, 32.9, 33.95, 35, 36.07, 37.13, 38.2, 39.28, 40.35, 41.43]])
    plot_dir = r'C:\Users\godin\Documents\VAN_ex\fig'
    plotly_3d_cams_compare(my_dws, kitti_dws, plot_dir, save=True, title="Title")

