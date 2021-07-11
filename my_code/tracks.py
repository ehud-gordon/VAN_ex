import numpy as np

import os
import pickle

import kitti, my_plot, tracks, utils, triang, features
TRACKS_PATH = r'C:\Users\godin\Documents\VAN_ex\tracks'
BITS_TRACKS_PER_IMG = 11


class Track:
    def __init__(self, id, pc,left_meas, right_meas, length, cam_id, match_id, prev=None, next=None):
        self.id = id
        self.orig_cam_id, self.orig_m_id = track_id_to_cam_match_idx(track_id=id)
        self.cam_id = cam_id
        self.match_id = match_id
        self.pc = pc
        self.left_x = left_meas[0]; self.left_y = left_meas[1]
        self.right_x = right_meas[0]; self.right_y = right_meas[1]
        self.length = length
        self.prev = prev
        self.next = next

    def __repr__(self):
        message = (
            f'id={self.id}, pc={self.pc}, left=({self.left_x:.1f}, {self.left_y:.1f})'
            f', right=({self.right_x:.2f}, {self.right_y:.2f}), cam_id={self.cam_id}, length={self.length}'
            f', orig=({self.orig_cam_id},{self.orig_m_id}), prev={self.prev}, next={self.next}'
        )
        return message


class Tracks_DB:
    def __init__(self, td=None, frames_idx=None):
        self.td = dict() if td is None else td
        self.frames_idx = [] if frames_idx is None else frames_idx

    def add_frame(self, matches_li_lj, i,j, kp_li, kp_ri, kp_lj, kp_rj, pc_lr_i_in_li, pc_lr_j_in_lj=None):
        if not self.frames_idx: self.frames_idx.append(i)
        self.frames_idx.append(j)
        assert kp_li.shape[1] == kp_ri.shape[1] == pc_lr_i_in_li.shape[1]
        assert kp_lj.shape[1] == kp_rj.shape[1]
        if pc_lr_j_in_lj is not None:
            assert kp_lj.shape[1] == kp_rj.shape[1] == pc_lr_j_in_lj.shape[1]        
        self.td[j] = dict()
        if i not in self.td:
            self.td[i] = dict()
        trains = [m.trainIdx for m in matches_li_lj] #
        # Below indicates a bug where two different kps in li (e.g. queries[63,64]=115,116) are matched to same kp in lj (trains[63:65]=91,91)
        if np.sum(np.bincount(trains) >= 2) > 0:
            print(f"add_frame({j}) DOUBLE ERROR")
        for match in matches_li_lj:
            match_i_id = match.queryIdx
            match_j_id = match.trainIdx
            pc_i_meas_in_li = pc_lr_i_in_li[:, match_i_id] # the pc estimate from frame i
            if pc_lr_j_in_lj is not None:
                pc_j_meas_in_lj = pc_lr_j_in_lj[:, match_j_id] # the pc estimate from frame j, should be real close to pc_li
            else:
                pc_j_meas_in_lj = 0
            lj_meas = kp_lj[:, match_j_id]
            rj_meas = kp_rj[:, match_j_id]
            # add all tracks that extend existing previous tracks
            if match_i_id in self.td[i]: # re-use existing track
                prev_track = self.td[i][match_i_id]
                prev_track.next = (j, match_j_id)
                track_id = prev_track.id
                track_length = prev_track.length + 1
                new_track_j = Track(id=track_id, pc=pc_j_meas_in_lj, left_meas=lj_meas, right_meas=rj_meas, length=track_length,
                                     cam_id=j, match_id=match_j_id, prev=(i, match_i_id))
                if match_j_id in self.td[j]:
                    weird = self.td[j][match_j_id]
                    print(f"error1, {new_track_j}")
                self.td[j][match_j_id]= new_track_j
            else: # add new tracks
                track_id = cam_match_idx_to_track_id(i, match_i_id)
                li_meas = kp_li[:,match_i_id]
                ri_meas = kp_ri[:, match_i_id]
                new_track_i = Track(id=track_id, pc=pc_i_meas_in_li, left_meas=li_meas, right_meas=ri_meas, length=1, cam_id=i,
                                     match_id=match_i_id, next=(j, match_j_id))
                self.td[i][match_i_id] = new_track_i

                new_track_j = Track(id=track_id, pc=pc_j_meas_in_lj, left_meas=lj_meas, right_meas=rj_meas, length=2, cam_id=j,
                                     match_id=match_j_id, prev=(i, match_i_id))
                if match_j_id in self.td[j]:
                    weird2 = self.td[j][match_j_id]
                    print("error2", new_track_j)
                self.td[j][match_j_id] = new_track_j

    def get_track_ids(self, cam_id):
        try:
            return {track.id for track in self.td[cam_id].values()} # set of ints
        except KeyError:
            return None

    def get_tracks(self, cam_id):
        try:
            return list(self.td[cam_id].values())  # list of tuples
        except KeyError:
            return None
    
    def get_track(self, cam_id, match_id):
        try:
            self.td[cam_id][match_id]
        except KeyError:
            return None

    def serialize(self, dir_path, title="tracks"):
        d = dict()
        d['td'] = self.td
        d['frames_idx'] = self.frames_idx
        pkl_path = os.path.join(dir_path, f'{title}.pkl')
        utils.clear_path(pkl_path)
        with open(pkl_path, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return pkl_path

def cam_match_idx_to_track_id(cam_id, match_id):
    return (cam_id << BITS_TRACKS_PER_IMG) + match_id  # 2050 = 1<<11 + 2

def track_id_to_cam_match_idx(track_id):
    cam_id = track_id >> BITS_TRACKS_PER_IMG # 2050 = 1
    match_id = track_id % (1<<BITS_TRACKS_PER_IMG) # 2050 = 2
    return cam_id, match_id

def read(tracks_pkl_path):
        with open(tracks_pkl_path, 'rb') as handle:
            d = pickle.load(handle)
        tracks_db = Tracks_DB(td=d['td'], frames_idx=d['frames_idx'])
        return tracks_db

def re_serialize(pkl_path):
    pkl_dir, name, ext = utils.dir_name_ext(pkl_path)
    print(pkl_dir, name, ext)
    with open(pkl_path, 'rb') as handle:
        d = pickle.load(handle)
    print(d.keys())
    d['ext_l0_to_li_s'] = d.pop('ext_l0_to_lj_s')
    with open(pkl_path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

def my_output_results(out_path, ext_l0_to_li_s):
    frames_idx = list(range(0, 501))
    ext_li_to_l0_s = utils.inv_extrinsics_mult(ext_l0_to_li_s)
    import results
    # output important plots and stats
    rots_total_error, trans_total_error = results.output_results(out_path, ext_li_to_l0_s, frames_idx, "stage2", 0, plot=True, save=True)
    # create folder
    stage2_dir =  os.path.join(out_path, 'stage2' + f'_{trans_total_error:.1f}_{rots_total_error:.1f}')
    utils.make_dir_if_needed(stage2_dir)

def visualize_tracks_frame(tracks_pkl_path, i):
    tracks_db = read(tracks_pkl_path)
    kp_l = []
    kp_r = []
    pc_in_l0 = []
    for track in tracks_db.get_tracks(i):
        kp_l.append((track.left_x, track.left_y))
        kp_r.append((track.right_x, track.right_y))
        pc_in_l0.append(track.pc)
    kp_l = np.array(kp_l).T
    kp_r = np.array(kp_r).T
    pc_in_l0 = np.array(pc_in_l0).T
    img_li, img_ri = kitti.read_images(i)
    my_plot.plotly_kp_pc_inliers(img_li, img_ri, kp_l, kp_r, pc_in_l0, np.ones(pc_in_l0.shape[1], dtype=bool), title=f"tracks_vis_L0_CS_f={i}",save=False, plot=True)
    ext_l0_to_li = tracks_db.ext_l0_to_li_s[i]
    ext_li_to_l0 = utils.inv_extrinsics(ext_l0_to_li)
    k, ext_id, ext_l_to_r = kitti.read_cameras()
    pc_in_li = triang.triang(kp_l,kp_r, k, ext_id, ext_l_to_r)
    import time; time.sleep(3)
    my_plot.plotly_kp_pc_inliers(img_li, img_ri, kp_l, kp_r, pc_in_li, np.ones(pc_in_li.shape[1], dtype=bool), title=f"tracks_vis_Li_CS_f={i}",save=False, plot=True)
    # draw = features.DrawMatchesDouble(img_li, img_ri, kp_l, kp_r, pc=pc_in_li, i=i)
    # draw.draw_matches_double()
    pc_in_li_to_l0 = np.vstack((pc_in_li, np.ones(pc_in_li.shape[1])))
    pc_in_li_to_l0 = ext_li_to_l0[:3] @ pc_in_li_to_l0
    import time; time.sleep(3)
    my_plot.plotly_kp_pc_inliers(img_li, img_ri, kp_l, kp_r, pc_in_li_to_l0, np.ones(pc_in_li.shape[1], dtype=bool),title=f"tracks_vis_in_Li_to_l0_CS_f={i}", save=False, plot=True)
    a = 3





if __name__=="__main__":
    i, m_id = track_id_to_cam_match_idx(4508351)
    print(i, m_id)

    tracks_pkl_path = r'C:\Users\godin\Documents\VAN_ex\out\07-05-15-43_relq_forest_099_011_global_2760\stage2_58.7_120.1\stage2_tracks_2760_filtered.pkl'
    tracks_db = read(tracks_pkl_path)
    i=2201
    visualize_tracks_frame(tracks_pkl_path, i)
    a=3

    print('finished')

