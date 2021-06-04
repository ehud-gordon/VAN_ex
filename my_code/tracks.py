import numpy as np

import os
import pickle
import utils
TRACKS_PATH = r'C:\Users\godin\Documents\VAN_ex\tracks'
MAX_TRCK_PER_IMG = 1024
class Tracks:
    def __init__(self, endframe, td=None, ext_l1s=None):
        self.endframe = endframe
        if td is None:
            self.td = dict() # tracks dict
            self.td[0] = dict()
        else:
            self.td = td
        if ext_l1s is None:
            ext_l1 = np.diag((1,1,1,1))
            self.ext_l1s = [ext_l1]
        else:
            self.ext_l1s = ext_l1s # takes points in l_{i-1} and transforms them to l_i

    def add_frame(self, matches_l0_l1, l1_idx, kp_l0, kp_r0, kp_l1, kp_r1, ext_l1, pc_l0_r0, pc_l1_r1):
        kp_l0 = kp_l0.astype(int)
        kp_r0 = kp_r0.astype(int)
        self.ext_l1s.append(ext_l1)
        kp_l1 = kp_l1.astype(int); kp_r1 = kp_r1.astype(int)
        l0_idx = l1_idx-1
        self.td[l1_idx] = dict()
        for m in matches_l0_l1:
            l0_m_idx = m.queryIdx
            l1_m_idx = m.trainIdx
            pc_l0 = pc_l0_r0[:, l0_m_idx]
            pc_l1 = pc_l1_r1[:, l1_m_idx]
            # add all matches to l1
            if l0_m_idx in self.td[l0_idx]: # re-use existing track
                prev_track = self.td[l0_idx][l0_m_idx]
                track_id = prev_track[0]
                track_length = prev_track[3]
                #                               0           1                            2                      3           4
                self.td[l1_idx][l1_m_idx]= [track_id, tuple(kp_l1[:,l1_m_idx]), tuple(kp_r1[:,l1_m_idx]), track_length+1, pc_l1]
            else: # create new track
                track_id = (l0_idx << 10) + l0_m_idx # 1025 = 1<<10 + 1
                self.td[l0_idx][l0_m_idx] = [track_id, tuple(kp_l0[:,l0_m_idx]), tuple(kp_r0[:, l0_m_idx]),1, pc_l0]
                self.td[l1_idx][l1_m_idx] = [track_id, tuple(kp_l1[:, l1_m_idx]), tuple(kp_r1[:, l1_m_idx]),2, pc_l1]
        self.kp_l0 = kp_l1
        self.kp_r0 = kp_r1


    def get_track_ids(self, camera_id):
        try:
            return {l[0]%MAX_TRCK_PER_IMG for l in self.td[camera_id].values()} # set of ints
        except KeyError:
            return None

    def get_tracks(self, camera_id):
        try:
            return list(self.td[camera_id].values())  # list of tuples
        except KeyError:
            return None

    def get_pixels(self,caemra_id, track_id):
        try:
            return self.td[caemra_id][track_id][2:4]
        except KeyError:
            return None
    
    def get_track(self, caemra_id, track_id):
        try:
            self.td[caemra_id][track_id]
        except KeyError:
            return None

    def serialize(self):
        d = dict()
        d['td'] = self.td
        d['endframe'] = self.endframe
        d['ext_l1s'] = self.ext_l1s
        path = os.path.join(utils.track_path(),utils.get_time_path()+f'_{self.endframe}.pickle')
        with open(path, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read(path):
        with open(path, 'rb') as handle:
            d = pickle.load(handle)
        tracks_db = Tracks(td=d['td'], ext_l1s=d['ext_l1s'],
                      endframe=d['endframe'])
        return tracks_db
