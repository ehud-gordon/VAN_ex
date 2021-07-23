""" plot functions """
import numpy as np
import plotly.graph_objects as go
import os, webbrowser
from utils import sys_utils

def plotly_save_fig(fig, title="", plot_dir="", save=False, plot=True):
    fig.update_layout(title_text=title, title_x=0.5, font=dict(size=16))
    if save:
        path = os.path.join(plot_dir, title + '.html')
        fig.write_html(path, auto_open=False)
        json_dir = os.path.join(plot_dir, 'JSON'); sys_utils.make_dir_if_needed(json_dir)
        json_path = os.path.join(json_dir, title +'.JSON')
        fig.write_json(json_path)
        if plot: webbrowser.open(sys_utils.path_to_windows(path), new=2)
    if not save and plot:
        path = sys_utils.get_avail_path("tmp.html")
        fig.write_html(path, auto_open=False)
        webbrowser.open(sys_utils.path_to_windows(path), new=2)

def plot_3D_cams(dws_names_colors, frames_idx, title="", plot_dir="", save=True, plot=False, add_sliders=True):
    num_frames = len(frames_idx)
    num_traces = len(dws_names_colors)

    fig = go.Figure()

    if num_frames > 1000:
        inds = np.arange(0, num_frames, 10)
        num_frames = inds.size
        dws_names_colors = [(dws[:,inds], label, color) for dws,label,color in dws_names_colors]
        frames_idx = frames_idx[inds]
    
    # create scatters
    pose_traces, marker_traces = [], []
    for dw, name, color in dws_names_colors:
        pose_trace = go.Scatter3d(x=dw[0], y=dw[2], z=dw[1], name=name, mode='markers+lines', line=dict(color=color),
                                marker=dict(size=3.5, color=color, opacity=0.5),
                                hovertemplate="(%{x:.1f}, %{z:.1f}, %{y:.1f}) frame=%{text:.0f}",
                                text=frames_idx, legendgroup=f"g{name}")
        pose_traces.append(pose_trace)
        if not add_sliders: continue
        marker_trace = go.Scatter3d(x=[dw[0,0]],  y=[dw[2,0]],  z=[dw[1,0]], 
                            mode="markers", legendgroup=f'g{name}', showlegend=False,
                            marker=dict(color=color),name=name,
                            hovertemplate="(%{x:.1f},%{z:.1f},%{y:.1f})" + "<br>frame=%{text:.0f}",
                                          text=[frames_idx[0]])
        marker_traces.append(marker_trace)
    fig.add_traces(pose_traces)
    if add_sliders: fig.add_traces(marker_traces)

    camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.2, y=-2, z=0.5) );fig.update_layout(scene_camera=camera)
    
    if add_sliders:
        # Create and add a slider
        dws_list = [t[0] for t in dws_names_colors] # list of #num_traces objects, where dws_list[i] is the (3,n) dws associated with trace i
        prev_x_list = [dws[0] for dws in dws_list]; prev_y_list = [dws[2] for dws in dws_list]; prev_z_list = [dws[1] for dws in dws_list]
        steps = []
        for i in range(num_frames):
            step = dict(method="update",  args=
                            [{
                            "x":prev_x_list + [[x_dws[min(len(x_dws)-1,i)]] for x_dws in prev_x_list],
                            "y": prev_y_list + [[y_dws[min(len(y_dws)-1,i)]] for y_dws in prev_y_list],
                            "z": prev_z_list + [[z_dws[min(len(z_dws)-1,i)]] for z_dws in prev_z_list],
                            "text": [frames_idx]*num_traces + [ [frames_idx[min(i, len(x_dws)-1)]] for x_dws in prev_x_list ]
                            }],
                            label=f"{frames_idx[i]}")
            steps.append(step)
        sliders = [dict(active=0, currentvalue={"prefix": "Frame="}, steps=steps)]
        fig.update_layout(sliders=sliders)
    
    fig.update_layout(showlegend=True)
    fig.update_scenes(zaxis_autorange="reversed", xaxis_title='X', yaxis_title='Z', zaxis_title='Y', aspectmode='data')
    fig.update_layout(title_text=title, title_x=0.5, font=dict(size=14))
    plotly_save_fig(fig, f'3D_cams_{title}', plot_dir, save, plot)

def scatter(y, x=None, title="", plot_dir="", mode='lines+markers', xaxis="Frames", yaxis="Y", yrange=None, save=True, plot=False):
    scatter = go.Scatter(x=x, y=y, mode=mode)
    fig = go.Figure(scatter)    
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    if yrange: fig.update_yaxes(range=yrange)
    plotly_save_fig(fig, title, plot_dir, save, plot)

def scatters(name_y_dict, x=None, title="", plot_dir="", mode='lines+markers', xaxis="Frames", yaxis="Y", yrange=None, save=True, plot=False):
    fig = go.Figure()
    for name, y in name_y_dict.items():
        fig.add_trace(go.Scatter(x=x,y=y, name=name, mode=mode))
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    if yrange: fig.update_yaxes(range=yrange)
    plotly_save_fig(fig, title, plot_dir, save, plot)
