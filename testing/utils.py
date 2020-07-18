def almost_equal(x, y, places=7):
    return round(abs(x - y), places) == 0

def _adjlist_equal(adj1, adj2):
    if len(adj1) != len(adj2):
        return False
    for v in range(len(adj1)):
        if len(adj1[v]) != len(adj2[v]):
            return False
        if len(adj1[v]) == 0:
            continue
        for w in range(len(adj1[v])):
            if adj1[v][w] != adj2[v][w]:
                return False
    return True


def assert_graphs_equal(graph1, graph2):
    assert _adjlist_equal(graph1.fadjlist, graph2.fadjlist)