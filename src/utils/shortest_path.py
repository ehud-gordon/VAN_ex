""" Utility function with scipy shortest path """

def shortest_path(source, target, predecessors_from_source, only_len=False):
    """ compute shortest path from source to target.

    :param source: index of source
    :param target: index of target
    :param predecessors_from_source: result of dijkstra(indices=source)
    :return: list of indices [target, ... , source]. if source=5 and target=0, returns [0, 1, 3, 4, 5]
    """
    assert target < len(predecessors_from_source) and source < len(predecessors_from_source)
    if target==source:
        if only_len: return 0
        return []
    path_length = 0
    res = [target]
    start=target
    while start != source:
        next_ = predecessors_from_source[start]
        path_length +=1
        if not only_len:
            res.append(next_)
        start = next_
    if only_len:
        return path_length
    return res