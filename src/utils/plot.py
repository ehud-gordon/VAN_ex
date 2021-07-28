""" plot functions """

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
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
        path = sys_utils.get_avail_path(os.path.join(os.getcwd(),"tmp.html"))
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

def scatter(y, x=None, title="", plot_dir="", mode='lines+markers', xaxis="frame", yaxis="Y", yrange=None, save=True, plot=False):
    scatter = go.Scatter(x=x, y=y, mode=mode)
    fig = go.Figure(scatter)    
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    if yrange: fig.update_yaxes(range=yrange)
    plotly_save_fig(fig, title, plot_dir, save, plot)

def scatters(name_y_dict, x=None, title="", plot_dir="", mode='lines+markers', xaxis="frame", yaxis="Y", yrange=None, save=True, plot=False):
    fig = go.Figure()
    for name, y in name_y_dict.items():
        fig.add_trace(go.Scatter(x=x,y=y, name=name, mode=mode))
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    if yrange: fig.update_yaxes(range=yrange)
    plotly_save_fig(fig, title, plot_dir, save, plot)

def concat_df(keyframes, arr, total_df, stage_name):
    keyframes = np.array(keyframes)[:,None]
    arr = np.asarray(arr)
    arr_df = pd.DataFrame( np.hstack([keyframes, arr.reshape(-1, 1)]), columns=['frame', 'value'])
    arr_df['stage'] = stage_name
    total_df = pd.concat([total_df, arr_df])
    return total_df

def concat_rot_trans(stage_name, keyframes, rot, total_rot, trans, total_trans):
    keyframes = np.array(keyframes)[:,None]
    
    rot = pd.DataFrame( np.hstack([keyframes, rot.reshape(-1, 1)]), columns=['frame', 'value'])
    rot['stage'] = stage_name
    new_total_rot = pd.concat([total_rot, rot])

    trans = pd.DataFrame(np.hstack([keyframes, trans]), columns=['frame', 'x', 'y', 'z'])
    trans['stage'] = stage_name
    trans = pd.melt(trans, id_vars=['frame', 'stage'], value_vars=['x', 'y', 'z'], var_name="axis", value_name="value")
    new_total_trans = pd.concat([total_trans, trans])

    return new_total_rot, new_total_trans

def plot_rot(rot_df, title="", plot_dir="", plot=False, save=True):
    fig = px.line(rot_df, x='frame', y='value', color="stage", labels={'value':'rot diff'},
                  color_discrete_sequence=["red","blue","green","orange"])
    # fig.update_traces(mode="lines")
    fig.update_layout(hovermode="x unified")
    plotly_save_fig(fig, title=title+' Rotation Error', plot_dir=plot_dir, save=save, plot=plot)

def plot_trans(trans_df, title="", plot_dir="", plot=False, save=True):
    colors = ["red","blue","green","orange", "purple"]
    stages = trans_df['stage'].unique()
    stage_color_map = {stage:color for stage, color in zip(stages, colors)}
    fig = px.line(trans_df, x='frame', y='value', facet_row='axis', color="stage", labels={'value':'meters'},
                  color_discrete_sequence=colors)
    # fig.update_traces(mode="lines")
    fig.update_layout(hovermode="x unified")
    plotly_save_fig(fig, title=title+' Translation Error', plot_dir=plot_dir, save=save, plot=plot)
    # plot L2 norm
    norm_df = trans_df.groupby(["stage","frame"])["value"].apply(lambda ser: np.linalg.norm(ser.to_numpy())).reset_index()
    fig = px.line(norm_df, x='frame', y='value', color="stage", labels={'value':'L2 Norm (meters)'},
                 color_discrete_map=stage_color_map)
    # fig.update_traces(mode="lines")
    fig.update_layout(hovermode="x unified")
    plotly_save_fig(fig, title=title+' Translation Error L2 Norm', plot_dir=plot_dir, save=save, plot=plot)

def plot_rot_trans(rot_df, trans_df, title="", plot_dir="", plot=False, save=True):
    plot_rot(rot_df, title, plot_dir, plot, save)
    plot_trans(trans_df, title, plot_dir, plot, save)

def rot_trans_stats(out_dir, keyframes, stage_name, rot_abs, trans_abs, rot_rel, trans_rel):
    num_frames = keyframes[-1]
    avg_rot_abs = np.sum(rot_abs) / num_frames
    avg_trans_abs = np.sum(trans_abs,axis=0) / num_frames # (3,)
    avg_trans_abs_norm = np.sum(avg_trans_abs)

    avg_rot_rel = np.sum(rot_rel) / num_frames
    avg_trans_rel = np.sum(trans_rel,axis=0) / num_frames # (3,)
    avg_trans_rel_norm = np.sum(avg_trans_rel)
    stats =[f'{stage_name}', 
            f'avg. absolute rotation error:  {np.sum(rot_abs):.2f} / {num_frames} = {avg_rot_abs:.2f} degrees',
            f'avg. absolute translation error: x-axis: {avg_trans_abs[0]:.2f}, y-axis: {avg_trans_abs[1]:.2f}, z-axis: {avg_trans_abs[2]:.2f} meters',
            f'avg. absolute translation error total: {np.sum(trans_abs):.2f} / {num_frames} = {avg_trans_abs_norm:.2f} meters',
            '\n'
            f'avg. relative rotation error: {np.sum(rot_rel):.2f} / {num_frames} = {avg_rot_rel:.2f} degrees',
            f'avg. relative translation error: x-axis: {avg_trans_rel[0]:.2f}, y-axis: {avg_trans_rel[1]:.2f}, z-axis: {avg_trans_rel[2]:.2f} meters',
            f'avg. relative translation error total: {np.sum(trans_rel)} / {num_frames} = {avg_trans_rel_norm:.2f} meters\n']

    with open(os.path.join(out_dir, f'stats_{stage_name}.txt'), 'w') as f:
                f.writelines('\n'.join(stats))
