import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import gridspec
import scipy
import os, webbrowser

import utils
from utils import rund, und_title

############ MATPLOTLIB ############
def plt_diff_trans_vecs(trans_diff, plot_dir,title, frames_idx=None, save=True, plot=False):
    """ :param trans_diff: (3,n)  
        :frames_idx: list/array of size n+1
    """
    if frames_idx:
        sizes = np.array(frames_idx)[1:] - np.array(frames_idx)[:-1]
        trans_diff /= sizes
    else:
        frames_idx = np.arange(trans_diff.shape[1]+1)
    startframe = frames_idx[0]; endframe = frames_idx[-1]
    plt.figure()
    plt.plot(frames_idx[1:], trans_diff[0], label="tx")
    plt.plot(frames_idx[1:], trans_diff[1], label="ty")
    plt.plot(frames_idx[1:], trans_diff[2], label="tz")
    plt.ylabel("meters"); plt.xlabel("frames")
    plt.title (f"Relative abs diff btwn my and kitti trans vectors, {title} in [{startframe}-{endframe}]")
    plt.legend()
    if save:
        path = os.path.join(plot_dir, f"diff_trans{und_title(title)}{endframe}.png")
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()

def plt_diff_rot_matrices(rot_diff, plot_dir, title, frames_idx=None, save=True, plot=False):
    """ :param rot_diff: list/array of size n
        :frames_idx ndarray of size n+1
    """
    if frames_idx:
        sizes = np.array(frames_idx)[1:] - np.array(frames_idx)[:-1]
        rot_diff /= sizes
    else:
        frames_idx = np.arange(len(rot_diff)+1)
    startframe = frames_idx[0]; endframe = frames_idx[-1]
    plt.figure()
    plt.plot(frames_idx[1:], rot_diff)
    plt.ylabel("degrees"); plt.xlabel("Frames")
    plt.title (f"Relative diff btwn my and kitti rotation matrices, {title} frames [{startframe}-{endframe}")
    if save:
        path = os.path.join(plot_dir, f"diff_rot{und_title(title)}{endframe}.png")
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()

def plt_2D_cams(dws_labels_colors, title, plot_dir="plt", save=True, plot=False):
    """ :param dws_labels_colors: list [(2d_arr, scatter_name), ..., ()]  """
    plt.figure()
    for dws, label,color in dws_labels_colors:
        plt.scatter(x=dws[0], y=dws[2], label=label, color=color)
    plt.xlabel('x'); plt.ylabel('z')
    plt.title(f'left camera location {title}')
    if len(dws_labels_colors) > 1:
        plt.legend()
    if save:
        path = os.path.join(plot_dir, f'plt_2D_cams_{title}' + '.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()
def plt_3D_cams(dws_labels_colors, title, plot_dir="plt",save=True, plot=False):
    fig = plt.figure()
    xmin = ymin = zmin = np.inf; xmax = ymax = zmax = -np.inf
    ax = fig.add_subplot(projection='3d')
    for dws, label, color in dws_labels_colors:
        ax.scatter(dws[0], dws[2], dws[1], label=label, color=color)
        xmin, ymin, zmin = np.min( np.column_stack(([xmin, ymin, zmin], dws)) ,axis=1)
        xmax, ymax, zmax = np.max( np.column_stack(([xmax, ymax, zmax], dws)) ,axis=1)
    ax.set_xlim([xmin - 1, xmax + 1])
    ax.set_ylim([zmin -5, zmax + 5])  # not a mistake, plt's Y axis is our Z-Axis
    ax.set_zlim([ymin - 1, ymax + 1])  # not a mistake, plt's z-axis is our Y-axis
    ax.invert_zaxis()  # not a mistake, - plt's z axis is our Y axis
    ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y')  # not a mistake
    plt.title(f"left camera location {title}")
    if len(dws_labels_colors):
        plt.legend()
    if save:
        path = os.path.join(plot_dir, f'plt_3D_cams_{title}' + '.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()

def plt_bundle_errors(errors_before, errors_after, idx, title="", plot_dir="plt", save=True, plot=False,xlabel="frames"):
    plt.figure()
    plt.plot(idx, errors_before, label="before")
    plt.plot(idx, errors_after, label="after")
    plt.xlabel(xlabel)
    plt.ylabel('error')
    plt.title(f'Graph errors before and after optimization {title}')
    plt.legend()
    if save:
        path = os.path.join(plot_dir, f'0_{idx[-1]}_graph_errors{und_title(title)}' + '.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()

class Draw_KP_PC_INLIERS:
    def __init__(self, img_l, kp_l, kp_r, pc, inliers_bool, i=0):
        assert kp_l.shape[1] == kp_r.shape[1] == pc.shape[1] == len(inliers_bool)
        self.i=i
        self.num_points = kp_l.shape[1]
        self.img_l = img_l
        self.kp_l = kp_l
        self.kp_r = kp_r
        self.pc = pc
        self.inliers_bool = inliers_bool
        self.ckdtree_kp = scipy.spatial.cKDTree(kp_l.T)
        self.ckdtree_pc = scipy.spatial.cKDTree(pc.T)
        self.cur_pc = None
        self.cur_kp = None
        self.cur_i = 0

    def on_hover(self, event):
        if event.inaxes is None:
            return
        if event.inaxes == self.ax_l:
            closest_index = self.ckdtree_kp.query([event.xdata,event.ydata])[1]
        else:
            x,y,z = self.getxyz(event)
            closest_index = self.ckdtree_pc.query([x, z, y])[1]
        self.cur_i = closest_index
        self.draw_match(closest_index)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def onkeypress(self, event):
        if event.key == 'c':  # clear all
            self.clear()
        if event.key == 'p':  # draw only next
            self.cur_i = (self.cur_i-1 % self.num_points)
            self.draw_match(self.cur_i)
        if event.key == 'n':  # draw only next
            self.cur_i = (self.cur_i + 1 % self.num_points)
            self.draw_match(self.cur_i)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def getxyz(self,event):
        pressed = self.ax_pc.button_pressed
        self.ax_pc.button_pressed = -1  # some value that doesn't make sense.
        coords = self.ax_pc.format_coord(event.xdata, event.ydata)  # coordinates string in the form x=value, y=value, z= value
        self.ax_pc.button_pressed = pressed
        coords_str = [s.split('=')[1].strip() for s in coords.split(',')]
        res = []
        for s in coords_str:
            try:
                f = float(s)
            except ValueError: # matplotlib uses non-ASCII char for '-', which raises an exception when calling float()
                f = -float(s[1:])
            res.append(f)
        return res

    def draw_match(self, i):
        self.clear()
        closest_pix = self.kp_l[:, i]  # (2,)
        closest_3d = self.pc[:, i]  # (3,)
        self.cur_kp = self.ax_l.scatter(x=closest_pix[0], y=closest_pix[1], s=40, color='blue')
        self.cur_pc = self.ax_pc.scatter(closest_3d[0], closest_3d[2], closest_3d[1], s=100, color='blue')

    def clear(self):
        if self.cur_kp:
            try:
                self.cur_kp.remove()
            except:
                pass
        if self.cur_pc:
            try:
                self.cur_pc.remove()
            except:
                pass

    def draw_figure(self, title="", save=False, plot_dir=""):
        self.fig = plt.figure(figsize=plt.figaspect(2.))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 3])
        in_bool = self.inliers_bool
        # image
        self.ax_l = self.fig.add_subplot(gs[0])
        self.ax_l.imshow(self.img_l, cmap="gray")
        # kp scatter
        self.ax_l.scatter(x=self.kp_l[0,in_bool], y=self.kp_l[1,in_bool], s=5, color="green", marker='o')
        self.ax_l.scatter(x=self.kp_l[0, ~in_bool], y=self.kp_l[1, ~in_bool], s=5, color="red",marker='o')
        self.ax_l.xaxis.tick_top()
        # PC scatter
        self.ax_pc = self.fig.add_subplot(gs[1], projection='3d')
        self.ax_pc.scatter(self.pc[0, in_bool], self.pc[2,in_bool], self.pc[1,in_bool], label="pc", color="green") # not a mistake, - plt's z axis is kitti Y axis
        self.ax_pc.scatter(self.pc[0,~in_bool], self.pc[2, ~in_bool], self.pc[1, ~in_bool], label="pc",color="red")  # not a mistake, - plt's z axis is kitti Y axis
        self.ax_pc.invert_zaxis()  # not a mistake, - kitti's y axis is reversed
        self.ax_pc.set_xlabel('X'); self.ax_pc.set_ylabel('Z'); self.ax_pc.set_zlabel('Y')  # not a mistake

        self.fig.subplots_adjust(top=0.94, bottom=0.0, left=0.015, right=0.975, hspace=0.0, wspace=0.0)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        # self.fig.canvas.mpl_connect('button_press_event', self.on_hover)
        self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        plt.suptitle(f"frame={self.i} {title}")
        if save:
            path = os.path.join(plot_dir, f'plt_kp_pc_inliers_{self.i}{utils.lund(title)}' + '.png')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.show()


        plt.show()
############ PLOTLY ############
def plotly_save_fig(fig, title="", plot_dir="", save=True, plot=True):
    fig.update_layout(title_text=title, title_x=0.5, font=dict(size=14))
    if save:
        path = os.path.join(plot_dir, title + '.html')
        fig.write_html(path, auto_open=False)
        json_dir = os.path.join(plot_dir, 'JSON'); utils.make_dir_if_needed(json_dir)
        json_path = os.path.join(json_dir, title +'.JSON')
        fig.write_json(json_path)
        png_dir = os.path.join(plot_dir,'PNG'); utils.make_dir_if_needed(png_dir)
        png_path = os.path.join(png_dir, title + '.png')
        fig.write_image(png_path)
        if plot: webbrowser.open(utils.path_to_windows(path), new=2)
    if not save and plot:
        path = utils.get_avail_path("tmp.html")
        fig.write_html(path, auto_open=False)
        webbrowser.open(utils.path_to_windows(path), new=2)

def plotly_bar(y, bins=None, title="", plot_dir="", save=True, plot=False):
    if bins is None:
        bins = np.arange(len(y))
    fig = go.Figure(go.Bar(x=bins, y=y))
    fig.data[0].text = y
    fig.update_traces(textposition='inside', textfont_size=12)
    fig.update_layout(bargap=0, title_text=title, title_x=0.5, font=dict(size=18))
    fig.update_traces(marker_color='blue', marker_line_color='blue', marker_line_width=1)

    plotly_save_fig(fig, f'bar{und_title(title)}', plot_dir, save, plot)
    
def plotly_hist(y, bins=None, title="", plot_dir="", density=True, save=True, plot=True):
    if bins is None:
        bins = np.arange(np.max(y)+2)
    y_hist, bins = np.histogram(y, bins=bins, density=density)
    plotly_bar(y=y_hist, bins=bins, title=title, plot=plot, plot_dir=plot_dir)

def plotly_2D_cams(dws_names_colors, title, plot_dir="", frames_idx=None, save=True, plot=False):
    num_frames = dws_names_colors[0][0].shape[1] # 11
    num_traces = len(dws_names_colors)
    frames_idx = np.array(frames_idx) if frames_idx else np.arange(0,num_frames)  # [0-10]

    if num_frames > 1000:
        inds = np.arange(0, num_frames, 10)
        num_frames = inds.size
        dws_names_colors = [(dws[:,inds], label, color) for dws,label,color in dws_names_colors]
        frames_idx = frames_idx[inds]

    fig = go.Figure()
    dws_list = [t[0] for t in dws_names_colors]
    prev_x_list = [dws[0] for dws in dws_list]; prev_y_list = [dws[2] for dws in dws_list]

    # create fixed scatters
    pose_traces, marker_traces = [], []
    for dws, name, color in dws_names_colors:
        pose_trace = go.Scatter(x=dws[0], y=dws[2], mode='lines+markers', marker=dict(size=3.5, color=color),
                          name=name, line=dict(color=color, width=1),
                          hovertemplate="x:%{x:.1f}, z:%{y:.1f}, frame=%{text:.0f}", text=frames_idx,
                          legendgroup=f"g{name}")
                          
        marker_trace = go.Scatter(x=[dws[0,0]],  y=[dws[2,0]], 
                            mode="markers", legendgroup=f'g{name}', showlegend=False,
                            marker=dict(color="blue"),name=name,
                            hovertemplate="(%{x:.1f},%{y:.1f})" + "<br>frame=%{text:.0f}",
                                          text=[frames_idx[0]])
        pose_traces.append(pose_trace); marker_traces.append(marker_trace)
    fig.add_traces(pose_traces); fig.add_traces(marker_traces)
    

    
    # Create and add slider
    steps = []
    for i in range(num_frames):
        step = dict(method="update",  args=[
                {"x":prev_x_list + [[x_dws[i]] for x_dws in prev_x_list],
                    "y": prev_y_list + [[y_dws[i]] for y_dws in prev_y_list],
                "text": [frames_idx]*num_traces +[[frames_idx[i]] for _ in range(num_traces)]} ], label=f"{frames_idx[i]}")
        steps.append(step)

    sliders = [dict(active=0, currentvalue={"prefix": "Frame="}, steps=steps)]
    fig.update_layout(sliders=sliders)

    
    fig.update_layout(xaxis=dict(zeroline=False), xaxis_title="X", yaxis=dict(zeroline=False), yaxis_title="Z")
    fig.update_yaxes(scaleanchor = "x",scaleratio = 1)
    fig.update_layout(title_text=title, title_x=0.5, font=dict(size=14))


    plotly_save_fig(fig, f'2D_cams_{title}', plot_dir, save, plot)

def plotly_3D_cams(dws_names_colors, title, plot_dir="", frames_idx=None, save=True, plot=False):
    num_frames = dws_names_colors[0][0].shape[1] # 11
    num_traces = len(dws_names_colors)
    frames_idx = np.array(frames_idx) if frames_idx is not None else np.arange(0,num_frames)  # [0-10]

    fig = go.Figure()

    if num_frames > 1000:
        inds = np.arange(0, num_frames, 10)
        num_frames = inds.size
        dws_names_colors = [(dws[:,inds], label, color) for dws,label,color in dws_names_colors]
        frames_idx = frames_idx[inds]
    
    dws_list = [t[0] for t in dws_names_colors]
    prev_x_list = [dws[0] for dws in dws_list]; prev_y_list = [dws[2] for dws in dws_list]; prev_z_list = [dws[1] for dws in dws_list]
    
    # create scatters
    pose_traces, marker_traces = [], []
    for dw, name, color in dws_names_colors:
        pose_trace = go.Scatter3d(x=dw[0], y=dw[2], z=dw[1], name=name, mode='markers+lines', line=dict(color=color),
                                marker=dict(size=3.5, color=color, opacity=0.5),
                                hovertemplate="(%{x:.1f}, %{z:.1f}, %{y:.1f}) frame=%{text:.0f}",
                                text=frames_idx, legendgroup=f"g{name}")
        marker_trace = go.Scatter3d(x=[dw[0,0]],  y=[dw[2,0]],  z=[dw[1,0]], 
                            mode="markers", legendgroup=f'g{name}', showlegend=False,
                            marker=dict(color="blue"),name=name,
                            hovertemplate="(%{x:.1f},%{z:.1f},%{y:.1f})" + "<br>frame=%{text:.0f}",
                                          text=[frames_idx[0]])
        pose_traces.append(pose_trace); marker_traces.append(marker_trace)
    fig.add_traces(pose_traces); fig.add_traces(marker_traces)

    camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.2, y=-2, z=0.5) );fig.update_layout(scene_camera=camera)
    
    # Create and add a slider
    steps = []
    for i in range(num_frames):
        step = dict(method="update",  args=[
                        {"x":prev_x_list + [[x_dws[i]] for x_dws in prev_x_list],
                        "y": prev_y_list + [[y_dws[i]] for y_dws in prev_y_list],
                        "z": prev_z_list + [[z_dws[i]] for z_dws in prev_z_list],
                        "text": [frames_idx]*num_traces +[[frames_idx[i]] for _ in range(num_traces)]} ], label=f"{frames_idx[i]}")
        steps.append(step)
    sliders = [dict(active=0, currentvalue={"prefix": "Frame="}, steps=steps)]
    fig.update_layout(sliders=sliders)
    
    # fig.update_layout(margin=dict(l=0, r=0, t=40))
    fig.update_layout(showlegend=True)
    fig.update_scenes(zaxis_autorange="reversed", xaxis_title='X', yaxis_title='Z', zaxis_title='Y', aspectmode='cube')
    fig.update_layout(title_text=title, title_x=0.5, font=dict(size=14))
    plotly_save_fig(fig, f'3D_cams_{title}', plot_dir, save, plot)
    fig.update_scenes(aspectmode='data')
    plotly_save_fig(fig, f'3D_cams_axes_equal_{title}', plot_dir, save, plot)

def plotly_scatter(y, x=None, mode='lines+markers', title="", plot_dir="", xaxis="Frames", yaxis="Y", yrange=None, save=True, plot=False):
    scatter = go.Scatter(x=x, y=y, mode=mode)
    fig = go.Figure(scatter)    
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    if yrange: fig.update_yaxes(range=yrange)
    plotly_save_fig(fig, title, plot_dir, save, plot)

def plotly_scatters(name_y_dict, x=None, title="", plot_dir="", mode='lines+markers', xaxis="Frames", yaxis="Y", yrange=None, save=True, plot=False):
    fig = go.Figure()
    for name, y in name_y_dict.items():
        fig.add_trace(go.Scatter(x=x,y=y, name=name, mode=mode))
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    if yrange: fig.update_yaxes(range=yrange)
    plotly_save_fig(fig, title, plot_dir, save, plot)
    
def get_k_small(arr,k, min=True):
    if not min:
        arr = -arr
    inds = np.argpartition(arr,k)[:k]
    sort_inds = np.argsort(arr[inds])
    orig_sort_inds = inds[sort_inds]
    vals_sort = arr[orig_sort_inds]
    if not min:
        vals_sort *= -1
    return orig_sort_inds, vals_sort

def plotly_k_scatter(llsd, frames_idx, title="", plot_dir="",plot=False, save=True, k=5, yaxis="Y", min=True, kitti=False):
    fig = go.Figure()
    vals_frames_idx = []
    vals_arr, inds_arr = [], []
    rot_diffs_arr, trans_diffs_arr = [], []
    for l in llsd:
        # extract k smallest/largest values
        d = l[0]
        cur_frame = d['cur_frame']
        vals_frames_idx.append(cur_frame)
        vals = np.array(d['y'])
        inds_k, vals_k  = get_k_small(vals, k, min=min)
        vals_arr.append(vals_k); inds_arr.append(inds_k)
        if kitti: 
            rot_diffs = np.array(d['rot_diffs']); trans_diffs = np.array(d['trans_diffs'])
            rot_diffs_arr.append(rot_diffs[inds_k]); trans_diffs_arr.append(trans_diffs[inds_k])
    vals_arr = np.asarray(vals_arr).T # (k, num_of_frames)
    inds_arr = np.asarray(inds_arr).T # (k, num_of_frames)
    if kitti: rot_diffs_arr = np.asarray(rot_diffs_arr).T; trans_diffs_arr = np.asarray(trans_diffs_arr).T; 
    f = np.array(frames_idx)
    correct_inds_arr = [f[row] for row in inds_arr]
    if kitti:
        res = []
        for ind_row, rot_row, trans_row in zip(correct_inds_arr, rot_diffs_arr, trans_diffs_arr):
            new_txt = [f'f={ind}, rot={rot:.2f}, trans={trans:.2f}  ' for ind,rot,trans in zip(ind_row, rot_row, trans_row)]
            res.append(new_txt)
        correct_inds_arr = res
    # add scatters 
    i=0
    for vals_i, inds_txt_i in zip(vals_arr, correct_inds_arr):
        mode='markers+lines' if i==0 else 'markers'
        fig.add_trace( go.Scatter(x=vals_frames_idx, y=vals_i, mode=mode, marker=dict(size=8), name=f'{i}', hovertemplate='%{text}, %{y:.2f}', text=inds_txt_i))
        i+=1
    # add lines
    for frame, f_vals in zip(vals_frames_idx, vals_arr.T):
        fig.add_trace(go.Scatter( x=[frame,frame], y=[f_vals[0], f_vals[-1]], line=dict(color="black"), showlegend=False, hoverinfo='skip'))
    
    fig.update_layout(xaxis_title="frames", yaxis_title=yaxis)
    fig.update_yaxes(range=[0,1])
    fig.update_layout(hovermode="x unified")
    plotly_save_fig(fig, f'{title}', plot_dir, save, plot)

def plot_trans_rot_norms(rot_norms, trans_norms, title="", plot_dir="", frames_idx=None, save=True, plot=False):
    num_frames = len(rot_norms) # 11
    frames_idx = frames_idx if frames_idx else np.arange(0,num_frames)  # [0-10]
    rot_scatter = go.Scatter(x=frames_idx, y=rot_norms, mode='lines', name="rot")
    trans_scatter = go.Scatter(x=frames_idx, y=trans_norms, mode='lines', name="trans")
    fig = go.Figure([rot_scatter, trans_scatter])
    name = f'rot_trans{und_title(title)}'
    fig.update_layout(xaxis_title="frames", yaxis_title="deg or meters")
    fig.update_layout(hovermode="x")

    plotly_save_fig(fig, name, plot_dir, save, plot)

def frames_slider_plot(llsd, title="", plot_dir="", xaxis="Frames", yaxis="", save=True, plot=False, yrange=None):
    """
    :param: llsd: list_of_lists_of_scatter_dcts
                  [l1,..., ln] where each li is a list of [d1,..] where dj = {'x':[], 'y':[], 'name':str, 'mode':mode, 'cur_frame':str() }
    """
    if not llsd: return
    num_frames = len(llsd)
    fig = go.Figure()
    l_of_lens = [len(l) for l in llsd]
    lens_cumsum = np.cumsum(l_of_lens)
    lens_cumsum = np.insert(lens_cumsum,0,0)
    # add scatters
    for l in llsd:
        for d in l:
            scat = go.Scatter(x=d['x'], y=d['y'], mode=d['mode'], name=d['name'], visible=False)
            fig.add_trace(scat)
    for i in range(l_of_lens[0]):
        fig.data[i].visible=True

    # Create and add slider
    steps = []
    for i in range(num_frames):
        cur_frame = llsd[i][0]['cur_frame']
        vises = [False] * len(fig.data)
        vises[(lens_cumsum[i]):(lens_cumsum[i+1])] = [True] * l_of_lens[i]
        step = dict(method="update", args=[{"visible": vises}], label=f"{cur_frame}")
        steps.append(step)

    sliders = [dict(active=0, currentvalue={"prefix": "Frame="}, steps=steps)]
    fig.update_layout(sliders=sliders)
    fig.update_layout(showlegend=True)


    fig.update_layout(title_text=title, title_x=0.5, font=dict(size=14))
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    if yrange: fig.update_yaxes(range=yrange)

    plotly_save_fig(fig, title, plot_dir, save, plot)

def plotly_pnp_inliers(img_l, img_r, kp_l, kp_r,pc, kp_l_unmatched, kp_r_unmatched, inliers_bool, title="", plot_dir="", plot=False, save=True):
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        specs=[
                            [{"type":"xy"}],
                            [{"type":"xy"}],
                        ],
                        horizontal_spacing = 0.01,
                        vertical_spacing=0.01)
    # IMAGES TRACES
    img_l_trace  = go.Heatmap(z=img_l, colorscale='gray',name="img_l")
    img_r_trace  = go.Heatmap(z=img_r, colorscale='gray',name="img_r")

    fig.add_traces([img_l_trace, img_r_trace], rows=[1, 2], cols=[1, 1])

    fig.update_yaxes(autorange='reversed', scaleanchor='x', visible=False, constrain='domain', showgrid=False, zeroline=False, row=1, col=1)
    fig.update_xaxes(constrain='domain',showgrid=False, visible=False, zeroline=False, row=1, col=1)

    fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain', visible=False, showgrid=False, zeroline=False, row=2, col=1)
    fig.update_xaxes(constrain='domain',showgrid=False, zeroline=False, visible=False, row=2, col=1)

    fig.data[0].showscale = False; fig.data[0].coloraxis = None
    fig.data[1].showscale = False; fig.data[1].coloraxis = None


    # KEYPOINTS TRACES
    kp_l_inliers = kp_l[:, inliers_bool]; kp_l_inliers_text = [f'({x:.1f}, {y:.1f})' for x, y in kp_l_inliers.T]

    kp_l_outliers = kp_l[:, ~inliers_bool]; kp_l_outliers_text = [f'({x:.1f}, {y:.1f})' for x, y in kp_l_outliers.T]

    kp_r_inliers = kp_r[:, inliers_bool]; kp_r_inliers_text = [f'({x:.1f}, {y:.1f})' for x, y in kp_r_inliers.T]

    kp_r_outliers = kp_r[:, ~inliers_bool]; kp_r_outliers_text = [f'({x:.1f}, {y:.1f})' for x, y in kp_r_outliers.T]

    inliers_pc = pc[:, inliers_bool]; inliers_pc_txt = [f'({x:.1f}, {y:.1f}, {z:.1f})' for x, y, z in inliers_pc.T]

    outliers_pc = pc[:, ~inliers_bool]; outliers_pc_txt = [f'({x:.1f}, {y:.1f}, {z:.1f})' for x, y, z in outliers_pc.T]

    kp_l_inliers_tt = [f'pc={x}<br>kp_r={y}' for x,y in zip(inliers_pc_txt, kp_r_inliers_text)]
    kp_l_inliers_scat = go.Scatter(x=kp_l_inliers[0], y=kp_l_inliers[1], mode="markers", name="kp_l_inliers",
                             marker=dict(color="green", size=9), hovertemplate="(%{x:.1f}, %{y:.1f}), %{text}", text=kp_l_inliers_tt)
    kp_l_outliers_tt = [f'pc={x}<br>kp_r={y}' for x, y in zip(outliers_pc_txt, kp_r_outliers_text)]
    kp_l_outliers_scat = go.Scatter(x=kp_l_outliers[0], y=kp_l_outliers[1], mode="markers", name="kp_l_outliers",
                              marker=dict(color="red", size=9), hovertemplate="(%{x:.1f}, %{y:.1f}), %{text}",text=kp_l_outliers_tt)
    fig.add_traces([kp_l_inliers_scat, kp_l_outliers_scat], rows=[1, 1], cols=[1, 1])

    kp_r_inliers_tt = [f'pc={x}<br>kp_l={y}' for x, y in zip(inliers_pc_txt, kp_l_inliers_text)]
    kp_r_inliers_scat = go.Scatter(x=kp_r_inliers[0], y=kp_r_inliers[1], mode="markers", name="kp_r_inliers",
                                   marker=dict(color="green", size=9), hovertemplate="(%{x:.1f}, %{y:.1f}), %{text}", text=kp_r_inliers_tt)
    kp_r_outliers_tt = [f'pc={x}<br>kp_l={y}' for x, y in zip(outliers_pc_txt, kp_l_outliers_text)]
    kp_r_outliers_scat = go.Scatter(x=kp_r_outliers[0], y=kp_r_outliers[1], mode="markers", name="kp_r_outliers",
                                    marker=dict(color="red", size=9), hovertemplate="(%{x:.1f}, %{y:.1f}), %{text}", text=kp_r_outliers_tt)
    fig.add_traces([kp_r_inliers_scat, kp_r_outliers_scat], rows=[2, 2], cols=[1, 1])

    # unmatched scatters
    l_unmatched_scat = go.Scatter(x=kp_l_unmatched[0], y=kp_l_unmatched[1], mode="markers", name="kp_l_unmatched", opacity=0.6, marker=dict(color="yellow", size=5))
    r_unmatched_scat = go.Scatter(x=kp_r_unmatched[0], y=kp_r_unmatched[1], mode="markers", name="kp_r_unmatched",opacity=0.6, marker=dict(color="yellow", size=5))
    # fig.add_traces([l_unmatched_scat, r_unmatched_scat], rows=[1,2], cols=[1,1])

    # GENERAL
    fig.update_layout(coloraxis_showscale=False, showlegend=False)
    plotly_save_fig(fig, title, plot_dir, save, plot)

def plotly_kp_pc_inliers(img_l, img_r, kp_l, kp_r, pc, inliers_bool, title="", plot_dir="", plot=False, save=True):
    fig = make_subplots(rows=2, cols=2,
                        specs=[
                            [{"type":"xy"},{"type":"xy"}],
                            [{"type":"scene", 'colspan':2}, None ]
                        ],
                        horizontal_spacing = 0.01,
                        vertical_spacing=0.01)
    # IMAGES TRACES
    img_l_trace  = go.Heatmap(z=img_l, colorscale='gray',name="img_l")
    img_r_trace  = go.Heatmap(z=img_r, colorscale='gray',name="img_r")

    fig.add_traces([img_l_trace, img_r_trace], rows=[1, 1], cols=[1, 2])

    fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain', showgrid=False, zeroline=False, row=1, col=1)
    fig.update_xaxes(constrain='domain',showgrid=False, zeroline=False, row=1, col=1)

    fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain', showgrid=False, zeroline=False, row=1, col=2)
    fig.update_xaxes(constrain='domain',showgrid=False, zeroline=False, row=1, col=2)

    fig.data[0].showscale = False; fig.data[0].coloraxis = None
    fig.data[1].showscale = False; fig.data[1].coloraxis = None


    # KEYPOINTS TRACES
    kp_l_inliers = kp_l[:, inliers_bool]; kp_l_inliers_text = [f'({x:.1f}, {y:.1f})' for x, y in kp_l_inliers.T]

    kp_l_outliers = kp_l[:, ~inliers_bool]; kp_l_outliers_text = [f'({x:.1f}, {y:.1f})' for x, y in kp_l_outliers.T]

    kp_r_inliers = kp_r[:, inliers_bool]; kp_r_inliers_text = [f'({x:.1f}, {y:.1f})' for x, y in kp_r_inliers.T]

    kp_r_outliers = kp_r[:, ~inliers_bool]; kp_r_outliers_text = [f'({x:.1f}, {y:.1f})' for x, y in kp_r_outliers.T]

    inliers_pc = pc[:, inliers_bool]; inliers_pc_txt = [f'({x:.1f}, {y:.1f}, {z:.1f})' for x, y, z in inliers_pc.T]

    outliers_pc = pc[:, ~inliers_bool]; outliers_pc_txt = [f'({x:.1f}, {y:.1f}, {z:.1f})' for x, y, z in outliers_pc.T]

    kp_l_inliers_tt = [f'pc={x}<br>kp_r={y}' for x,y in zip(inliers_pc_txt, kp_r_inliers_text)]
    kp_l_inliers_scat = go.Scatter(x=kp_l_inliers[0], y=kp_l_inliers[1], mode="markers", name="kp_l_inliers",
                             marker=dict(color="green"), hovertemplate="(%{x:.1f}, %{y:.1f}), %{text}", text=kp_l_inliers_tt)
    kp_l_outliers_tt = [f'pc={x}<br>kp_r={y}' for x, y in zip(outliers_pc_txt, kp_r_outliers_text)]
    kp_l_outliers_scat = go.Scatter(x=kp_l_outliers[0], y=kp_l_outliers[1], mode="markers", name="kp_l_outliers",
                              marker=dict(color="red"), hovertemplate="(%{x:.1f}, %{y:.1f}), %{text}",text=kp_l_outliers_tt)
    fig.add_traces([kp_l_inliers_scat, kp_l_outliers_scat], rows=[1, 1], cols=[1, 1])

    kp_r_inliers_tt = [f'pc={x}<br>kp_l={y}' for x, y in zip(inliers_pc_txt, kp_l_inliers_text)]
    kp_r_inliers_scat = go.Scatter(x=kp_r_inliers[0], y=kp_r_inliers[1], mode="markers", name="kp_r_inliers",
                                   marker=dict(color="green"), hovertemplate="(%{x:.1f}, %{y:.1f}), %{text}", text=kp_r_inliers_tt)
    kp_r_outliers_tt = [f'pc={x}<br>kp_l={y}' for x, y in zip(outliers_pc_txt, kp_l_outliers_text)]
    kp_r_outliers_scat = go.Scatter(x=kp_r_outliers[0], y=kp_r_outliers[1], mode="markers", name="kp_r_outliers",
                                    marker=dict(color="red"), hovertemplate="(%{x:.1f}, %{y:.1f}), %{text}", text=kp_r_outliers_tt)
    fig.add_traces([kp_r_inliers_scat, kp_r_outliers_scat], rows=[1, 1], cols=[2, 2])

    # PC TRACE
    inliers_pc_tt_txt = [f'kp_l={l}, kp_r={r}' for l,r in zip(kp_l_inliers_text, kp_r_inliers_text)]
    inliers_pc_trace = go.Scatter3d(x=inliers_pc[0], y=inliers_pc[2], z=inliers_pc[1], mode='markers',
                                    name="inliers_pc", marker=dict(color="green"),
                                    hovertemplate="(%{x:.1f}, %{z:.1f}, %{y:.1f})" + "<br>%{text}",
                                    text=inliers_pc_tt_txt)

    outliers_pc_tt_txt = [f'kp_l={l}, kp_r={r}' for l, r in zip(kp_l_outliers_text, kp_r_outliers_text)]
    outliers_pc_trace = go.Scatter3d(x=outliers_pc[0], y=outliers_pc[2], z=outliers_pc[1], mode='markers',
                                     name="outliers_pc", marker=dict(color="red"),
                                     hovertemplate="(%{x:.1f}, %{z:.1f}, %{y:.1f})" + "<br>%{text}",
                                     text=outliers_pc_tt_txt)

    fig.add_traces([inliers_pc_trace, outliers_pc_trace], rows=[2, 2], cols=[1, 1])

    fig.update_scenes(zaxis_autorange="reversed")
    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=0.2, y=-2, z=0.5))
    fig.update_layout(scene_camera=camera)
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Z', zaxis_title='Y'))

    # GENERAL
    # fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(showlegend=True)

    plotly_save_fig(fig, title, plot_dir, save, plot)

def plotly_inliers_outliers(img, kp, inliers_bool,  pc, title="", plot_dir="", plot=False, save=True):

    fig = make_subplots(rows=1, cols=2,
                        specs=[ [{"type":"xy"},{"type":"scene"}] ] )

    img_trace = go.Heatmap(z=img, colorscale='gray')
    fig.add_trace(img_trace, row=1, col=1)
    kp_text = [f'({x:.1f}, {y:.1f})' for x, y in kp.T]
    inliers_kp = kp[:, inliers_bool]; inliers_kp_text = [f'({x:.1f}, {y:.1f})' for x,y in inliers_kp.T]

    outliers_kp = kp[:,~inliers_bool]; outliers_kp_text = [f'({x:.1f}, {y:.1f})' for x,y in outliers_kp.T]

    inliers_pc = pc[:, inliers_bool]; inliers_pc_txt = [f'({x:.1f}, {y:.1f}, {z:.1f})' for x,y,z in inliers_pc.T]

    outliers_pc = pc[:, ~inliers_bool]; outliers_pc_txt = [f'({x:.1f}, {y:.1f}, {z:.1f})' for x, y, z in outliers_pc.T]

    inlier_scat = go.Scatter(x=inliers_kp[0], y=inliers_kp[1], mode="markers", name="inliers_kp", marker=dict(color="green"),
                             hovertemplate="(%{x:.1f}, %{y:.1f}), pc=%{text}",
                             text=inliers_pc_txt)
    outlier_scat = go.Scatter(x=outliers_kp[0], y=outliers_kp[1], mode="markers", name="outliers_kp", marker=dict(color="red"),
                              hovertemplate="(%{x:.1f}, %{y:.1f}), pc=%{text}",
                              text=outliers_pc_txt)
    fig.add_traces([inlier_scat, outlier_scat], rows=[1,1], cols=[1,1])


    inliers_pc_trace = go.Scatter3d(x=inliers_pc[0], y=inliers_pc[2], z=inliers_pc[1], mode='markers', name="inliers_pc", marker=dict(color="green"),
                               hovertemplate="(%{x:.1f}, %{y:.1f}, %{z:.1f})" + "<br>kp=%{text}",
                               text=inliers_kp_text)

    outliers_pc_trace = go.Scatter3d(x=outliers_pc[0], y=outliers_pc[2], z=outliers_pc[1], mode='markers',
                                    name="outliers_pc", marker=dict(color="red"),
                                    hovertemplate="(%{x:.1f}, %{y:.1f}, %{z:.1f})" + "<br>kp=%{text}",
                                    text=outliers_kp_text)

    fig.add_traces([inliers_pc_trace, outliers_pc_trace], rows=[1,1], cols=[2,2])

    fig.update_scenes(zaxis_autorange="reversed")
    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=0.2, y=-2, z=0.5))
    fig.update_layout(scene_camera=camera)
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Z', zaxis_title='Y'))


    fig.data[0].showscale=False
    fig.data[0].coloraxis=None

    fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain', row=1, col=1)
    fig.update_xaxes(constrain='domain', row=1, col=1)
    fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, row=1, col=1)

    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(showlegend=True)

    plotly_save_fig(fig, f'inliers_outliers{und_title(title)}', plot_dir, save, plot)
    
def plotly_cov_dets(cov_cj_cond_ci_dict, frames_idx, title="", plot_dir="", plot=False, save=True):
    # conditional
    num_frames = len(frames_idx) # 277
    cov_cj_cond_ci_s, cov_ci_cond_cj_s =[], []
    cj_cond_ci_dets, ci_cond_cj_dets =[], []
    for j in range(1,num_frames): # [1,..,276]
        cov_cj_cond_ci = cov_cj_cond_ci_dict[j][j-1]; cov_cj_cond_ci_s.append(cov_cj_cond_ci)
        cov_ci_cond_cj = cov_cj_cond_ci_dict[j-1][j]; cov_ci_cond_cj_s.append(cov_ci_cond_cj)
        cj_cond_ci_dets.append( np.linalg.det(cov_cj_cond_ci) )
        ci_cond_cj_dets.append( np.linalg.det(cov_ci_cond_cj) )
    title = rund(title)
    dct1 = {'det_lj_cond_li':cj_cond_ci_dets, 'det_li_cond_li':ci_cond_cj_dets}
    plotly_scatters(dct1, x=frames_idx, title=f"det_cond_{title}",plot_dir=plot_dir, yaxis="Det of cov matrix", save=save, plot=plot)
    
    # cumsum
    cumsum_cj_on_ci = utils.cumsum_mats(cov_cj_cond_ci_s); 
    cumsum_cj_on_ci_dets = [np.linalg.det(cov_mat) for cov_mat in cumsum_cj_on_ci]
    cumsum_ci_on_cj = utils.cumsum_mats(cov_ci_cond_cj_s)
    cumsum_ci_on_cj_dets = [np.linalg.det(cov_mat) for cov_mat in cumsum_ci_on_cj]
    dct1 = {'cumsum_lj_cond_li':cumsum_cj_on_ci_dets, 'cumsum_li_cond_lj':cumsum_ci_on_cj_dets}
    plotly_scatters(dct1, x=frames_idx, title=f"det_cumsum_cond_{title}",plot_dir=plot_dir, yaxis="Det of cov matrix", save=save, plot=plot)

def pose_graph_llsd(keyframes_idx, llsd_inliers, llsd_mahal, llsd_dets, title, plot_dir):
    if llsd_mahal:
        plotly_k_scatter(llsd_mahal, keyframes_idx, title=f"mahals_k_scatter_{title}", plot_dir=plot_dir, min=True, yaxis="Mahalanobis distance", plot=False, save=True)
        frames_slider_plot(llsd_mahal, title=f"mahal_dist_slider_{title}", plot_dir=plot_dir, yaxis="Mahalanobis distance", yrange=[0,5], plot=False, save=True)
    if llsd_inliers: 
        plotly_k_scatter(llsd_inliers, keyframes_idx, title=f"inliers_frac_k_scatter_{title}", plot_dir=plot_dir, min=False, yaxis="fraction of inliers", plot=False, save=True, kitti=True)
        frames_slider_plot(llsd_inliers, title=f"inliers_frac_slider_{title}", plot_dir=plot_dir, yaxis="fraction of inliers", yrange=[0,1], plot=False, save=True)         
    if llsd_dets:
        frames_slider_plot(llsd_dets, title=f"det_distance_{title}", plot_dir=plot_dir, yaxis="determinant distance", plot=False, save=True)

# if __name__=="__main__":
#     my_dws = np.array([[0, -0.004751, -0.01733, -0.02374, -0.02972, -0.03907, -0.05193, -0.06274, -0.07457, -0.09065,
#                         -0.113, -0.1339, -0.1606, -0.1888, -0.2123, -0.2353, -0.2611, -0.2875, -0.308, -0.3268, -0.3465,
#                         -0.3669, -0.3896, -0.4119, -0.4335, -0.4608, -0.4814, -0.4955, -0.517, -0.5386, -0.5611,
#                         -0.5823, -0.6074, -0.6329, -0.6569, -0.6807, -0.7041, -0.7271, -0.7565, -0.7859, -0.8093,
#                         -0.8333, -0.8557, -0.8819, -0.909, -0.9387, -0.9683, -0.9989, -1.031, -1.061, -1.093],[0, -0.01047, -0.02652, -0.03871, -0.05046, -0.06306, -0.08161, -0.09546, -0.1162, -0.1349,
#                         -0.1567, -0.1749, -0.1893, -0.2049, -0.2205, -0.2383, -0.2554, -0.2636, -0.2664, -0.271,
#                         -0.2803, -0.2863, -0.2884, -0.2893, -0.2887, -0.2915, -0.3015, -0.3129, -0.3239, -0.3437,
#                         -0.3679, -0.3938, -0.4187, -0.445, -0.4759, -0.5089, -0.5426, -0.577, -0.6097, -0.6417, -0.6742,
#                         -0.7039, -0.7336, -0.7621, -0.7896, -0.8157, -0.8407, -0.8659, -0.8908, -0.9154, -0.9394],[0, 0.5756, 1.154, 1.723, 2.291, 2.862, 3.444, 4.033, 4.628, 5.231, 5.842, 6.462, 7.104, 7.764,
#                         8.442, 9.128, 9.819, 10.52, 11.24, 11.99, 12.76, 13.55, 14.36, 15.18, 16.01, 16.86, 17.72, 18.6,
#                         19.5, 20.42, 21.35, 22.29, 23.24, 24.21, 25.19, 26.17, 27.17, 28.17, 29.19, 30.22, 31.25, 32.29,
#                         33.34, 34.39, 35.46, 36.52, 37.6, 38.68, 39.77, 40.86, 41.95]])
#     kitti_dws = np.array([[1.11e-16, 0.0035, 0.001174, -0.007832, -0.01508, -0.0204, -0.02969, -0.04153, -0.05594,
#                            -0.07008, -0.08798, -0.1055, -0.1239, -0.1445, -0.1645, -0.1845, -0.2077, -0.233, -0.2538,
#                            -0.2701, -0.2821, -0.2962, -0.3128, -0.3271, -0.3451, -0.3586, -0.3661, -0.3686, -0.3813,
#                            -0.3931, -0.4066, -0.4197, -0.4376, -0.4507, -0.4686, -0.4871, -0.5056, -0.5259, -0.5465,
#                            -0.5679, -0.5885, -0.6114, -0.6342, -0.6563, -0.68, -0.7046, -0.7318, -0.7509, -0.7637,
#                            -0.7861, -0.812],[0, -0.009789, -0.02241, -0.04141, -0.05923, -0.07423, -0.09078, -0.1099, -0.1348, -0.1576,
#                            -0.1803, -0.1997, -0.2188, -0.2366, -0.2543, -0.2738, -0.2924, -0.3036, -0.3092, -0.3152,
#                            -0.3207, -0.3252, -0.3292, -0.33, -0.3299, -0.3311, -0.3374, -0.3507, -0.3643, -0.3836,
#                            -0.4079, -0.4351, -0.4631, -0.4914, -0.5208, -0.5505, -0.5807, -0.6123, -0.6431, -0.6738,
#                            -0.7048, -0.7345, -0.7659, -0.7961, -0.8251, -0.8521, -0.8792, -0.8999, -0.9174, -0.9426,
#                            -0.9707], [2.22e-16, 0.5654, 1.128, 1.689, 2.251, 2.816, 3.39, 3.969, 4.556, 5.15, 5.751, 6.364, 6.99,
#                            7.638, 8.302, 8.978, 9.662, 10.35, 11.06, 11.8, 12.56, 13.34, 14.12, 14.93, 15.75, 16.58,
#                            17.43, 18.29, 19.17, 20.07, 20.99, 21.92, 22.88, 23.85, 24.82, 25.81, 26.79, 27.8, 28.8,
#                            29.82, 30.84, 31.87, 32.9, 33.95, 35, 36.07, 37.13, 38.2, 39.28, 40.35, 41.43]])
#     plot_dir = os.getcwd()
    # plotly_2D_cams_compare(my_dws, kitti_dws, plot_dir=plot_dir, save=True, title="Title")

