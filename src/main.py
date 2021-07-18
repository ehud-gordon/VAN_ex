import argparse
import os

import kitti
from stereo_slam import StereoSLAM
import utils.sys_utils as sys_utils

def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset_path", type=str, default=kitti.data_path(), help="path of kitti sequences and poses")
    parser.add_argument("--out_dir", type=str, default= sys_utils.out_dir(), help="path of output dir")
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--startframe", type=int, default=0)
    parser.add_argument("--endframe", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("-v", action="store_true", default=True, help="verbose")
    # args for Feature keypoints
    parser.add_argument("--detector", type=str, choices=["SIFT", "SURF", "AKAZE", "STAR"], default="SIFT", help="detector for keypoint detection")
    parser.add_argument("--descriptor", type=str, choices=["SIFT", "SURF", "AKAZE", "BRIEF"],default="SIFT", help="descriptor for keypoint detection")
    parser.add_argument("--matcher", type=str, choices=["BF", "FLANN"], default="BF")
    parser.add_argument("--feature_grid", action="store_true", default=False, help="boolean, whether to detect keypoints based on grid")
    # loop closure args
    parser.add_argument("--num_frames_between_lc", type=int, default=40, help="number of frames to wait between attempting Loop closure")
    parser.add_argument("--inliers_percent", type=float, default=0.6, help="required percent for consensus match in order to close loop")

    args = parser.parse_args()
    if args.endframe == 0:
        seq_length = kitti.get_seq_length(dataset_path=args.dataset_path) # 2761
        args.endframe = seq_length - 1 # 2760
    
    args.frames = list(range(args.startframe, args.endframe+1)) # list of frames indices

    # Get k, the camera matrix (intrinsics), and the pose from the left to right stereo camera
    k, ext_id, ext_l_to_r = kitti.read_cameras()


    args.out_path = os.path.join(args.out_dir, sys_utils.get_time_path())
    args.out_path += sys_utils.lund(args.run_name) + f'_{args.startframe}_{args.endframe}'
    if args.save or args.plot:
        sys_utils.clear_and_make_dir(args.out_path)


    return args, k, ext_l_to_r
if __name__=="__main__":
    args, k, ext_l_to_r = parse_args()
    slam = StereoSLAM(args, k, ext_l_to_r)
    slam.main()