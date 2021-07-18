import numpy as np
from dataclasses import dataclass

@dataclass
class StereoFeatures:
        idx: int  # frame index
        keypoints_left: np.ndarray
        descriptors_left: np.ndarray
        keypoints_right: np.ndarray
        pc : np.ndarray = None # point cloud


