import numpy as np


def cumsum_mats(list_of_mats):
    cumsum_arr = np.cumsum(list_of_mats, axis=0)
    cumsum_list = np.split(cumsum_arr, len(list_of_mats), axis=0)
    cumsum_list = [np.squeeze(mat) for mat in cumsum_list]
    return cumsum_list


def filter_(bool_array, *nd_arrays):
    """
    :param bool_array: boolean array of size n
    :param nd_arrays: ndarray of size (?,n)
    :return: the filtered arrays
    """
    for arr in nd_arrays:
        assert len(bool_array) == arr.shape[1]
    return [arr[:,bool_array] for arr in nd_arrays]


def get_perc_largest_indices(arr, perc):
    """
    find indices of the percent largest element in array
    :param arr: (n,) ndarray of numbers
    :param perc: number between [0-1], percentage of largest
    :return: boolean array, with True in place of largest perc
    """
    arr_abs = np.abs(arr)
    size = arr_abs.size
    num_of_largest = int(size * perc)
    idx_of_largest = np.argpartition(-arr_abs, num_of_largest)
    bool_array = np.zeros_like(arr, dtype=bool)
    bool_array[idx_of_largest[0:num_of_largest]] = True

    return bool_array