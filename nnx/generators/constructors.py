import numba
import numpy as np
import nnx

@numba.njit
def from_numpy_matrix(adjm, create_using):
    # constructs a graph from an adjacency matrix
    # If `adjm[i][j] != 0`, an edge `(i, j)` is inserted. `adjm` must be a square and symmetric matrix.
    if adjm.shape[0] != adjm.shape[1]:
        raise Exception('Simple undirected graphs need to be created using a square adjacency matrix')
    g = create_using()
    g.add_vertices(adjm.shape[0])
    triuadjm = np.triu(adjm)
    for i in range(adjm.shape[0]):
        for j in range(adjm.shape[1]):
            if triuadjm[i, j] != 0:
                g.add_edge(i, j)
    return g

@numba.njit
def from_edge_list(edge_list, create_using):
    #edge list of the form [(src, dst)]
    g = create_using()
    nvg = -1
    for e in edge_list:
        s, d = e
        maxv = max(s, d)
        if nvg < maxv:
            g.add_vertices(maxv - nvg)
            nvg = maxv
        g.add_edge(s, d)
    return g

