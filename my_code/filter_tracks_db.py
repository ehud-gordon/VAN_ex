import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px


import os

import kitti
import utils
import tracks
import my_plot
np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.4g" % x))

def eval_tracks_db(tracks_name):
    tracks_path = os.path.join(utils.track_path(), tracks_name+'.pickle')
    tracks_db = tracks.read(tracks_path)
    endframe = tracks_db.endframe  # 50
    k, ext_l0, ext_r0 = kitti.read_cameras()  # k=(3,4) ext_l0/r0 (4,4)
    left_x_diff = []; left_y_diff = []
    left_x_diff_baseline = []; left_y_diff_baseline = []
    right_x_diff = []; right_y_diff = []
    right_x_diff_baseline = []; right_y_diff_baseline = []
    num_baselines = 0
    num_others = 0
    tracks_lengths = []
    for cam_id in range(0, endframe):
        ext_li = kitti.read_poses_world_to_cam([cam_id])[0]
        proj_li = k @ ext_li
        ext_ri = ext_r0 @ ext_li
        proj_ri = k @ ext_ri
        for track in tracks_db.get_tracks(cam_id=cam_id):
            tracks_lengths.append(track.length)
            l0_meas = np.array([track.left_x, track.left_y])
            r0_meas = np.array([track.right_x, track.right_y])
            cur_pc = track.pc
            track_pc_orig = tracks_db.td[track.orig_cam_id][track.orig_m_id].pc
            track_pc_orig = np.hstack((track_pc_orig, [1]))
            # project orig_pc onto current cameras
            track_li_proj = proj_li @ track_pc_orig
            track_li_proj = track_li_proj[0:2] / track_li_proj[-1]
            track_ri_proj = proj_ri @ track_pc_orig
            track_ri_proj = track_ri_proj[0:2] / track_ri_proj[-1]
            l_diff = np.abs(l0_meas - track_li_proj)
            r_diff = np.abs(r0_meas - track_ri_proj)
            if track.orig_cam_id == cam_id:
                num_baselines += 1
                left_x_diff_baseline.append(l_diff[0]); left_y_diff_baseline.append(l_diff[1])
                right_x_diff_baseline.append(r_diff[0]); right_y_diff_baseline.append(r_diff[1])
            else:
                num_others += 1
                left_x_diff.append(l_diff[0]); left_y_diff.append(l_diff[1])
                right_x_diff.append(r_diff[0]); right_y_diff.append(r_diff[1])
    left_x_diff = np.array(left_x_diff); left_y_diff = np.array(left_y_diff)
    right_x_diff = np.array(right_x_diff); right_y_diff = np.array(right_y_diff)
    left_x_diff_baseline = np.array(left_x_diff_baseline); left_y_diff_baseline = np.array(left_y_diff_baseline)
    right_x_diff_baseline = np.array(right_x_diff_baseline); right_y_diff_baseline = np.array(right_y_diff_baseline)

    avg_left_x_diff_baseline = np.sum(left_x_diff_baseline) / num_baselines
    avg_left_y_diff_baseline = np.sum(left_y_diff_baseline) / num_baselines
    avg_right_x_diff_baseline = np.sum(right_x_diff_baseline) / num_baselines
    avg_right_y_diff_baseline = np.sum(right_y_diff_baseline) / num_baselines

    avg_left_x_diff = np.sum(left_x_diff) / num_others
    avg_left_y_diff = np.sum(left_y_diff) / num_others
    avg_right_x_diff = np.sum(right_x_diff) / num_others
    avg_right_y_diff = np.sum(right_y_diff) / num_others

    corrected_x_left_diff = avg_left_x_diff - avg_left_x_diff_baseline
    corrected_y_left_diff = avg_left_y_diff - avg_left_y_diff_baseline
    corrected_x_right_diff = avg_right_x_diff - avg_right_x_diff_baseline
    corrected_y_right_diff = avg_right_y_diff - avg_right_y_diff_baseline

    fig = px.box(left_x_diff, title="x coordinate diff", width=500, height=500)
    fig.update_layout(title_x=0.5)
    plot(fig)

    fig = px.box(left_y_diff, title="y coordinate diff", width=500, height=500)
    fig.update_layout(title_x=0.5)
    plot(fig)

def filter_tracks_db(tracks_name):
    print(f'filtering {tracks_name}')
    tracks_path = os.path.join(utils.track_path(), tracks_name+'.pickle')
    tracks_db = tracks.read(tracks_path)
    filtered_tracks_db = tracks.Tracks_DB(args=tracks_db.args, ext_l1s=tracks_db.ext_l1s)
    filtered_tracks_db.td[0] = dict()
    endframe = tracks_db.endframe  # 50
    k, ext_l0, ext_r0 = kitti.read_cameras()  # k=(3,4)
    filtered_tracks_ids = set()
    filtered_tracks_count = 0
    total_tracks = 0
    filtered_tracks_orig_frame_count = [0]
    for cam_id in range(1, endframe+1):
        filtered_tracks_orig_frame_count.append(0)
        filtered_tracks_db.td[cam_id] = dict()
        ext_li = kitti.read_poses_world_to_cam([cam_id])[0]
        ext_li = tracks_db.ext_l1s[cam_id]
        proj_li = k @ ext_li
        ext_ri = ext_r0 @ ext_li
        proj_ri = k @ ext_ri
        for track in tracks_db.get_tracks(cam_id=cam_id):
            total_tracks += 1
            if track.orig_cam_id == cam_id:
                continue
            if track.length > 2 and track.id in filtered_tracks_ids:
                filtered_tracks_count += 1
                continue
            l0_meas = np.array([track.left_x, track.left_y])
            r0_meas = np.array([track.right_x, track.right_y])
            track_pc_orig = tracks_db.td[track.orig_cam_id][track.orig_m_id].pc
            track_pc_orig = np.hstack((track_pc_orig, [1]))
            # project orig_pc onto current cameras
            track_li_proj = proj_li @ track_pc_orig ;track_li_proj = track_li_proj[0:2] / track_li_proj[-1]
            track_ri_proj = proj_ri @ track_pc_orig; track_ri_proj = track_ri_proj[0:2] / track_ri_proj[-1]
            l_diff = np.abs(l0_meas - track_li_proj)
            r_diff = np.abs(r0_meas - track_ri_proj)
            if np.sum(l_diff) > 6 or np.sum(r_diff) > 6:
                filtered_tracks_count += 1
                filtered_tracks_ids.add(track.id)
                continue
            # add new track
            filtered_tracks_db.td[track.cam_id][track.m_id] = track
            filtered_tracks_orig_frame_count[-1] += 1
            # check if orig in db, if not - add it as well
            if track.orig_m_id not in filtered_tracks_db.td[track.orig_cam_id]:
                filtered_tracks_db.td[track.orig_cam_id][track.orig_m_id] = tracks_db.td[track.orig_cam_id][track.orig_m_id]
                filtered_tracks_orig_frame_count[-2] += 0
    pct = filtered_tracks_count / total_tracks
    print(f"filtered {filtered_tracks_count}/{total_tracks}={pct:.2%} tracks")
    filtered_tracks_db.serialize(title=f"{tracks_name}_filtered")

    my_plot.plotly_bar(nums=filtered_tracks_orig_frame_count, title=f"filtered tracks original frame {tracks_name}")


if __name__=="__main__":
    dir, tracks_name, ext = utils.dir_name_ext("06_08_18_36_mine_global_50.pickle")
    filter_tracks_db(tracks_name=tracks_name)