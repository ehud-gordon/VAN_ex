import numpy as np

import os
import pickle
import my_code.utils as utils
TRACKS_PATH = r'C:\Users\godin\Documents\VAN_ex\tracks'

class Tracks:
    def __init__(self, kp_l0, kp_r0,first_frame_ind=0, td=None):
        self.first_frame_ind = first_frame_ind
        if td is None:
            self.td = dict()
            self.td[first_frame_ind] = dict()
        else:
            self.td = td
        self.kp_l0 = kp_l0.astype(int)
        self.kp_r0 = kp_r0.astype(int)


    def add_frame(self, matches_l0_l1, l1_idx, kp_l1, kp_r1):
        kp_l1 = kp_l1.astype(int); kp_r1 = kp_r1.astype(int)
        l0_idx = l1_idx-1
        self.td[l1_idx] = dict()
        for m in matches_l0_l1:
            l0_m_idx = m.queryIdx
            l1_m_idx = m.trainIdx
            # add all matches to l1
            if l0_m_idx in self.td[l0_idx]:
                track_id = self.td[l0_idx][l0_m_idx][0]
                track_length = self.td[l0_idx][l0_m_idx][3]
                self.td[l1_idx][l1_m_idx]= [track_id, tuple(kp_l1[:,l1_m_idx]), tuple(kp_r1[:,l1_m_idx]), track_length+1]
            else:
                track_id = (l0_idx, l0_m_idx)
                self.td[l0_idx][l0_m_idx] = [track_id, tuple(self.kp_l0[:,l0_m_idx]), tuple(self.kp_r0[:, l0_m_idx]),1]
                self.td[l1_idx][l1_m_idx] = [track_id, tuple(kp_l1[:, l1_m_idx]), tuple(kp_r1[:, l1_m_idx]),2]
        self.kp_l0 = kp_l1
        self.kp_r0 = kp_r1


    def get_track_ids(self, camera_id):
        try:
            return [l[0] for l in self.td[camera_id].values()] # list of lists
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
        d['kp_l0']=self.kp_l0
        d['kp_r0'] = self.kp_r0
        d['first_frame_ind'] = self.first_frame_ind
        path = os.path.join(TRACKS_PATH,utils.get_time_path()+'.pickle')
        with open(path, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read(path):
        with open(path, 'rb') as handle:
            d = pickle.load(handle)
        return Tracks(kp_l0=d['kp_l0'],kp_r0=d['kp_r0'], first_frame_ind=d['first_frame_ind'], td=d['td'])
