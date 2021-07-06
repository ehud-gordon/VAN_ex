import numpy as np

import argparse
import os, sys
import logging

import kitti, utils
from drive import Drive
from filter_tracks_db import filter_tracks_db

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
    parser.add_argument("--filt_type", type=str, choices=["rel", "rel_then_quant"], default="rel_then_quant")
    parser.add_argument("--quant_filt", type=float, default=0.99, help="quantile for filtering point cloud")
    parser.add_argument("--forest_cont", type=float, default=0.01, help="outliers percent for isolation forest")

    parser.add_argument("--endframe", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("-v", action="store_true", default=True, help="verbose")

    args = parser.parse_args()
    if not args.endframe:
        seq_length = kitti.get_seq_length(dataset_path=args.dataset_path) # 2761
        args.endframe = seq_length - 1 # 2760
    if args.v:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(message)s')

    args.out_path = os.path.join(args.out_dir, utils.get_time_path())
    args.out_path += '_' + utils.rund(args.run_name) +  ('relative' if args.relative else 'global') + f'_{args.endframe}'
    if args.store_tracks or args.save or args.plot:
        utils.clear_and_make_dir(args.out_path)


    return args
if __name__=="__main__":
    args = parse_args()
    drive = Drive(args=args)
    drive.main()
    import bundle
    filtered_tracks_db = filter_tracks_db(drive.stage2_tracks_path)
    ba = bundle.FactorGraphSLAM(filtered_tracks_db, args.out_path)
    ba.main()
    print('end')
