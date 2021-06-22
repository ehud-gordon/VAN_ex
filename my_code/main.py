import argparse

import numpy as np

import kitti
import utils
from drive import Drive
from filter_tracks_db import filter_tracks_db
import bundle

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.5g" % x))

def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset_path", type=str, default=kitti.data_path())
    parser.add_argument("--out_dir", type=str, default=utils.out_dir())

    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--relative", action="store_true", default=False)
    parser.add_argument("--kitti", action="store_true", default=False)

    parser.add_argument("--det", type=str, choices=["SIFT", "SURF", "AKAZE", "STAR"], default="SIFT", help="detector")
    parser.add_argument("--desc", type=str, choices=["SIFT", "SURF", "AKAZE", "BRIEF"],default="SIFT", help="descriptor")
    parser.add_argument("--feature_grid", action="store_true", default=False, help="boolean, whether to detect keypoints based on grid")
    parser.add_argument("--matcher", type=str, choices=["BF", "FLANN"], default="BF")
    parser.add_argument("--store_tracks", action="store_true", default=False)

    parser.add_argument("--endframe", type=int, default=0)

    args = parser.parse_args()
    if not args.endframe:
        seq_length = kitti.get_seq_length(dataset_path=args.dataset_path) # 2761
        args.endframe = seq_length - 1 # 2760

    return args
if __name__=="__main__":
    args = parse_args()
    drive = Drive(args=args)
    drive.main()
    filtered_tracks_db = filter_tracks_db(drive.tracks_db_path)
    ba = bundle.FactorGraphSLAM(tracks_path=filtered_tracks_db.path, tracks_db=filtered_tracks_db)
    ba.main()
    print('end')
