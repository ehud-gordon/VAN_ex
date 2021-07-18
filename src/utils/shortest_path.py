def shortest_path(source, target, predecessors_from_source):
    """ compute shortest path from source to target.

    :param source: index of source
    :param target: index of target
    :param predecessors_from_source: result of dijkstra(indices=source)
    :return: list of indices [target, ... , source]. if source=5 and target=0, returns [0, 1, 3, 4, 5]
    """
    assert target < len(predecessors_from_source) and source < len(predecessors_from_source)
    assert target != source
    if target==source: return []
    res = [target]
    start=target
    while start != source:
        next_ = predecessors_from_source[start]
        res.append(next_)
        start = next_
    return res