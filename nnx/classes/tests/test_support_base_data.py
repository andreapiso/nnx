import numba

#TODO: right now we do not have complete_graph classes

@numba.njit
def create_complete_graph_edge_list(n=3):
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            edge_list.append((i, j))
    return edge_list