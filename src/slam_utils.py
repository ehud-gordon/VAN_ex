import features

def match_two_pairs(i, n):
    featurez = Features()
    k, ext_id, ext_l_to_r = kitti.read_cameras()  # k=(3,4) ext_l0/r0 (4,4)
    kp_li, desc_li, kp_ri = featurez.get_kps_desc_stereo_pair(i)
    kp_li, kp_ri, pc_lr_i_in_li, desc_li = triang.triang_and_rel_filter(kp_li, kp_ri, k, ext_id, ext_l_to_r, desc_li)

    kp_ln, desc_ln, kp_rn = featurez.get_kps_desc_stereo_pair(n)
    kp_ln, kp_rn, pc_lr_n_in_ln, desc_ln = triang.triang_and_rel_filter(kp_ln, kp_rn, k, ext_id, ext_l_to_r, desc_ln)
    # match li-ln
    matches_li_ln = featurez.matcher.match(desc_li.T, desc_ln.T)  # list of matches [DMatch1,... DMatch1N]
    matches_li_ln = filter_matches(matches_li_ln, kp_li, kp_ln, is_stereo=False)

    return kp_li, kp_ri, pc_lr_i_in_li, kp_ln, kp_rn, pc_lr_n_in_ln, matches_li_ln