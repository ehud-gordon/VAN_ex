import numpy as np

import argparse
import os, sys
import logging

import kitti, utils
from drive import Drive
from tracks_analysis import filter_tracks_db

np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True, formatter=dict(float=lambda x: "%.5g" % x))

def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset_path", type=str, default=kitti.data_path())
    parser.add_argument("--out_dir", type=str, default=utils.out_dir())

    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--kitti", action="store_true", default=False)

    parser.add_argument("--det", type=str, choices=["SIFT", "SURF", "AKAZE", "STAR"], default="SIFT", help="detector")
    parser.add_argument("--desc", type=str, choices=["SIFT", "SURF", "AKAZE", "BRIEF"],default="SIFT", help="descriptor")
    parser.add_argument("--feature_grid", action="store_true", default=False, help="boolean, whether to detect keypoints based on grid")
    parser.add_argument("--matcher", type=str, choices=["BF", "FLANN"], default="BF")
    parser.add_argument("--store_tracks", action="store_true", default=False)
    parser.add_argument("--quant_filt", type=float, default=0.99, help="quantile for filtering point cloud")
    parser.add_argument("--forest_cont", type=float, default=0.01, help="outliers percent for isolation forest")
    parser.add_argument("--less_inliers", action="store_true", default=False)

    parser.add_argument("--startframe", type=int, default=0)
    parser.add_argument("--endframe", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("-v", action="store_true", default=True, help="verbose")

    args = parser.parse_args()
    
    if not args.endframe:
        seq_length = kitti.get_seq_length(dataset_path=args.dataset_path) # 2761
        args.endframe = seq_length - 1 # 2760
    
    args.frames_idx = list(range(args.startframe, args.endframe+1))
    
    # if args.v:
        # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(message)s')

    args.out_path = os.path.join(args.out_dir, utils.get_time_path())
    args.out_path += utils.lund(args.run_name) + f'_{args.startframe}_{args.endframe}'
    if args.store_tracks or args.save or args.plot:
        utils.clear_and_make_dir(args.out_path)


    return args
if __name__=="__main__":
    args = parse_args()
    drive = Drive(args=args)
    s2_ext_li_to_lj_s, s2_tracks_db = drive.main() # stage 2
    import bundle
    ba = bundle.FactorGraphSLAM(s2_ext_li_to_lj_s, s2_tracks_db, args.out_path)
    ba.main()
    print('end')
