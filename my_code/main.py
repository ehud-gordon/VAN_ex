import numpy as np
import argparse

import kitti
from drive import Drive

np.set_printoptions(edgeitems=30, linewidth=100000,suppress=True,
    formatter=dict(float=lambda x: "%.4g" % x))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=kitti.DATASET_5_PATH)

    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)

    parser.add_argument("--local", action="store_true", default=False)

    parser.add_argument("--det", type=str, choices=["SIFT", "SURF", "AKAZE"], default="SIFT",help="detector")
    parser.add_argument("--desc", type=str, choices=["SIFT", "SURF", "AKAZE",],default="SIFT", help="descriptor")
    parser.add_argument("--feature_grid", action="store_true", default=False, help="boolean, whether to detect based on grid")
    parser.add_argument("--matcher", type=str, choices=["BF", "FLANN"], default="BF")
    parser.add_argument("--store_tracks", action="store_true", default=False)

    parser.add_argument("--endframe", type=int, default=0)
    args = parser.parse_args()
    if not args.endframe:
        seq_length = kitti.get_seq_length(dataset_path=args.dataset_path) # 2761
        args.endframe = seq_length - 1 # 2760
    if args.endframe < 20:
        args.save = False
    return args

if __name__=="__main__":
    args = parse_args()
    drive = Drive(args=args)
    drive.main()
    print('end')
