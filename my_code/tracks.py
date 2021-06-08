import numpy as np

import os
import pickle
import utils
TRACKS_PATH = r'C:\Users\godin\Documents\VAN_ex\tracks'
BITS_TRACKS_PER_IMG = 11

class Track:
    def __init__(self, id, pc,left_meas, right_meas, length, cam_id, m_id):
        self.id = id
        self.orig_cam_id, self.orig_m_id = track_id_to_cam_m_idx(track_id=id)
        self.cam_id = cam_id; self.m_id = m_id
        self.pc = pc
        self.left_x = left_meas[0]; self.left_y = left_meas[1]
        self.right_x = right_meas[0]; self.right_y = right_meas[1]
        self.length = length

    def __repr__(self):
        message = (
            f'id={self.id}, pc={self.pc}, left=[{self.left_x:.2f},{self.left_y:.2f}], '
            f'right=[{self.right_x:.2f},{self.right_y:.2f}], cam_id={self.cam_id}, length={self.length}'
        )
        return message


class Tracks_DB:
    def __init__(self, args, td=None, ext_l1s=None):
        self.args = args
        self.endframe = args.endframe
        if td is None: # tracks dictionary
            self.td = dict() # tracks dict
            self.td[0] = dict()
        else:
            self.td = td
        if ext_l1s is None:
            ext_l1 = np.diag((1,1,1,1))
            self.ext_l1s = [ext_l1]
        else:
            self.ext_l1s = ext_l1s # takes points in l_{i-1} and transforms them to l_i

    def add_frame(self, matches_l0_l1, l1_id, kp_l0, kp_r0, kp_l1, kp_r1, ext_l1, pc_l0_r0, pc_l1_r1):
        self.ext_l1s.append(ext_l1)
        l0_id = l1_id-1
        self.td[l1_id] = dict()
        for m in matches_l0_l1:
            l0_m_id = m.queryIdx
            l1_m_id = m.trainIdx
            pc_l0 = pc_l0_r0[:, l0_m_id]
            pc_l1 = pc_l1_r1[:, l1_m_id]
            l1_meas = kp_l1[:, l1_m_id]
            r1_meas = kp_r1[:, l1_m_id]
            # add all matches to l1
            if l0_m_id in self.td[l0_id]: # re-use existing track
                prev_track = self.td[l0_id][l0_m_id]
                track_id = prev_track.id
                track_length = prev_track.length + 1
                self.td[l1_id][l1_m_id]= Track(id=track_id, pc=pc_l1, left_meas=l1_meas, right_meas=r1_meas,length=track_length, cam_id=l1_id, m_id=l1_m_id)
            else: # create new track
                track_id = cam_m_idx_to_track_id(cam_id=l0_id, m_id=l0_m_id)
                l0_meas = kp_l0[:,l0_m_id]
                r0_meas = kp_r0[:, l0_m_id]
                new_track_l0 = Track(id=track_id, pc=pc_l0, left_meas=l0_meas, right_meas=r0_meas,length=1, cam_id=l0_id, m_id=l0_m_id)
                self.td[l0_id][l0_m_id] = new_track_l0

                new_track_l1 = Track(id=track_id, pc=pc_l1, left_meas=l1_meas, right_meas=r1_meas, length=2, cam_id=l1_id, m_id=l1_m_id)
                self.td[l1_id][l1_m_id] = new_track_l1

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
    
    def get_track(self, cam_id, m_id):
        try:
            self.td[cam_id][m_id]
        except KeyError:
            return None

    def serialize(self, title=""):
        d = dict()
        d['td'] = self.td
        d['args'] = self.args
        d['ext_l1s'] = self.ext_l1s
        if not title:
            title=utils.get_time_path()
        path = os.path.join(utils.track_path(),f'{title}_{self.endframe}.pickle')
        path = utils.get_avail_path(path)
        with open(path, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

def cam_m_idx_to_track_id(cam_id, m_id):
    return (cam_id << BITS_TRACKS_PER_IMG) + m_id  # 2050 = 1<<11 + 2

def track_id_to_cam_m_idx(track_id):
    cam_id = track_id >> BITS_TRACKS_PER_IMG # 2050 = 1
    m_id = track_id % (1<<BITS_TRACKS_PER_IMG) # 2050 = 2
    return cam_id, m_id

def read(path):
        with open(path, 'rb') as handle:
            d = pickle.load(handle)
        tracks_db = Tracks_DB(td=d['td'], ext_l1s=d['ext_l1s'],
                      args=d['args'])
        return tracks_db
