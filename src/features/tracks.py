""" Database for storing stereo tracks """ 

import numpy as np

from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple

BITS_TRACKS_PER_IMG = 11 # hack to create great track ids

@dataclass
class StereoTrack:
    """ Track in stereo frames"""
    id: int # track id
    cam_id: int
    match_id: int
    point3: np.ndarray # the 3D location
    # the pixels location
    point2_left : np.ndarray
    point2_right: np.ndarray
    track_length : int
    prev_track: Tuple[int, int] = None # (prev_track_id, prev_track_m_id). Could be None
    next_track: Tuple[int, int] = None # (next_track_id, prev_track_m_id). Could be None

    def __post_init__(self):
        self.orig_cam_id, self.orig_match_id = track_id_to_cam_match_idx(self.id)
        self.point2_left_x = self.point2_left[0]
        self.point2_left_y = self.point2_left[1]
        self.point2_right_x = self.point2_right[0]
        self.point2_right_y = self.point2_right[1]

class StereoTracksDB:
    """ Database for storing tracks between frames of stereo images"""
    def __init__(self):
        self.db = defaultdict(dict) # database

    def __getitem__(self, item):
        return self.db[item]

    def add_frame(self, stereo_features_i, stereo_features_j, matches_i_j):
        """ add a set of tracks to the database. pixels are matched between frames i and j.

        :param stereo_features_i: StereoFeatures object (keypoints + point-cloud) of frame i
        :param stereo_features_j: StereoFeatures object (keypoints + point-cloud) of frame j
        :param matches_i_j: list of matches [DMatch1, ... , DMatchn] between features of frame i and j
        """
        i = stereo_features_i.idx
        keypoints_left_i = stereo_features_i.keypoints_left
        keypoints_right_i = stereo_features_i.keypoints_right
        pc_i = stereo_features_i.pc # point-cloud

        j = stereo_features_j.idx
        keypoints_left_j = stereo_features_j.keypoints_left
        keypoints_right_j = stereo_features_j.keypoints_right
        pc_j = stereo_features_j.pc  # point-cloud

        assert keypoints_left_i.shape[1] == keypoints_right_i.shape[1] == pc_i.shape[1]
        assert keypoints_left_j.shape[1] == keypoints_right_j.shape[1] == pc_j.shape[1]

        # add tracks to database
        for match in matches_i_j:
            match_i_id = match.queryIdx
            match_j_id = match.trainIdx
            point3_i = pc_i[:, match_i_id] # the point-cloud location from frame i
            point3_j = pc_j[:, match_j_id] # the point-cloud estimate from frame j
            point2_left_j = keypoints_left_j[:, match_j_id]
            point2_right_j = keypoints_right_j[:, match_j_id]
            # add all tracks that extend existing tracks
            if match_i_id in self.db[i]: # re-use existing track
                prev_track = self.db[i][match_i_id]
                prev_track.next_track = (j, match_j_id)
                track_id = prev_track.id
                track_length = prev_track.track_length + 1
                new_track_j = StereoTrack(id=track_id, point3=point3_j, point2_left=point2_left_j, point2_right=point2_right_j,
                                          track_length=track_length, cam_id=j, match_id=match_j_id, prev_track=(i, match_i_id))
                self.db[j][match_j_id]= new_track_j
            else: # Create new tracks
                # create track for frame i
                track_id = cam_match_idx_to_track_id(i, match_i_id)
                point2_left_i = keypoints_left_i[:, match_i_id]
                point2_right_i = keypoints_right_i[:, match_i_id]
                new_track_i = StereoTrack(id=track_id, point3=point3_i, point2_left=point2_left_i, point2_right=point2_right_i, 
                                          track_length=1, cam_id=i, match_id=match_i_id, next_track=(j, match_j_id))
                self.db[i][match_i_id] = new_track_i
                # create track for frame j
                new_track_j = StereoTrack(id=track_id, point3=point3_j, point2_left=point2_left_j, point2_right=point2_right_j,
                                          track_length=2, cam_id=j, match_id=match_j_id, prev_track=(i, match_i_id))
                self.db[j][match_j_id] = new_track_j

        return self

    def get_tracks(self, frame_idx):
        """ return all tracks from frame i"""
        try:
            return list(self.db[frame_idx].values())  # list of tuples
        except KeyError:
            return None

### Utility functions 
def cam_match_idx_to_track_id(cam_id, match_id):
    return (cam_id << BITS_TRACKS_PER_IMG) + match_id  # 2050 = 1<<11 + 2

def track_id_to_cam_match_idx(track_id):
    cam_id = track_id >> BITS_TRACKS_PER_IMG # 2050 = 1
    match_id = track_id % (1<<BITS_TRACKS_PER_IMG) # 2050 = 2
    return cam_id, match_id
