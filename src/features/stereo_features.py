import numpy as np
from dataclasses import dataclass

@dataclass
class StereoFeatures:
        idx: int
        keypoints_left: np.ndarray
        descriptors_left: np.ndarray
        keypoints_right: np.ndarray
        descriptors_right: np.ndarray
        pc : np.ndarray = None


x=np.array(2)
a = StereoFeatures(2,x,x,x,x)
print(a)

