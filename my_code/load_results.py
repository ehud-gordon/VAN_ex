import numpy as np
import os
import utils, results
import gtsam_utils as g_utils


frames_idx = list(range(2761))
stage2_pkl = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-10-13-49_0_2760/stage2_54.9_112.8/ext_li_to_lj_s_stage2_2760.pkl'
stage2_dir, _,_ = utils.dir_name_ext(stage2_pkl)
out_dir = os.path.dirname(stage2_dir)
ext_li_to_lj_s = utils.deserialize_ext_li_to_lj_s(stage2_pkl)
ext_li_to_l0_s = utils.concat_and_inv_ci_to_cj_s(ext_li_to_lj_s)
title="stage_2"
results.output_results(out_dir, ext_li_to_l0_s, frames_idx, title=title,start_time=0, save=True)

stage3_pkl = r'/mnt/c/users/godin/Documents/VAN_ex/out/07-10-13-49_0_2760/stage3_40.8_29.9/stage3_ext_lj_to_li_s_cond_covs_2760.pkl'
stage3_dir, _,_ = utils.dir_name_ext(stage3_pkl)
out_dir = os.path.dirname(stage3_dir)
ext_lj_to_li_s, _, frames_idx = g_utils.deserialize_bundle(stage3_pkl)
ext_li_to_l0_s = utils.concat_cj_to_ci_s(ext_lj_to_li_s)
title="stage_3"

results.output_results(out_dir, ext_li_to_l0_s, frames_idx, title=title,start_time=0, save=True)