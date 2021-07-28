import argparse

from utils import kitti
import stereo_slam
import utils.sys_utils as sys_utils


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset_path", type=str, default=kitti.data_path(), help="dir path of kitti sequences and poses")
    parser.add_argument("--out_dir", type=str, default= sys_utils.out_dir(), help="output dir path")
    parser.add_argument("--startframe", type=int, default=0)
    parser.add_argument("--endframe", type=int, default=0)
    # args for Feature keypoints
    parser.add_argument("--detector", type=str, choices=["SIFT", "SURF", "AKAZE", "STAR"], default="SIFT", help="detector for keypoint detection")
    parser.add_argument("--descriptor", type=str, choices=["SIFT", "SURF", "AKAZE", "BRIEF"],default="SIFT", help="descriptor for keypoint detection")
    parser.add_argument("--matcher", type=str, choices=["BF", "FLANN"], default="BF")
    parser.add_argument("--feature_grid", action="store_true", default=False, help="boolean, whether to detect keypoints based on grid")

    args = parser.parse_args()
    # if endframe==0, get all frames
    if args.endframe == 0:
        sequence_length = kitti.get_seq_length(dataset_path=args.dataset_path) # 2761
        args.endframe = sequence_length - 1 # 2760

    sys_utils.make_dir_if_needed(args.out_dir)
    
    args.frames = list(range(args.startframe, args.endframe+1)) # list of frames indices

    # Get k - the camera matrix (intrinsics), and the pose from the left to right stereo camera
    k, ext_id, ext_l_to_r = kitti.read_cameras() # (3,4) , (4,4), (4,4)
    
    return args, k, ext_l_to_r

if __name__=="__main__":
    args, k, ext_l_to_r = parse_args()
    slam = stereo_slam.StereoSLAM(args, k, ext_l_to_r)
    slam.main()