import numpy as np

import kitti
from drive import Drive

np.set_printoptions(edgeitems=30, linewidth=100000,suppress=True,
    formatter=dict(float=lambda x: "%.4g" % x))


if __name__=="__main__":
    drive = Drive(dataset_path=kitti.DATASET_5_PATH)
    drive.main()
    print('end')
