import numba
import numpy as np

@numba.njit
def pagerank(g, alpha=0.85, n=100, e=1.0e-6):
    """
    pagerank(g, alpha=0.85, n=100, e=1.0e-6)
    Calculate the [PageRank](https://en.wikipedia.org/wiki/PageRank) of the
    graph `g` parameterized by damping factor `alpha`, number of iterations 
    `n`, and convergence threshold `e`. Return a vector representing the
    centrality calculated for each node in `g`, or an error if convergence
    is not reached within `n` iterations.
    """
    nvv = g.nv
    a_div_outdegree = np.empty(nvv, dtype=np.float64)
    dangling_nodes = []
    for v in g.vertices:
        if g.outdegree(v) == 0:
            dangling_nodes.append(v)
            a_div_outdegree[v] = np.inf
        else:
            a_div_outdegree[v] = alpha / g.outdegree(v)
    x = np.full(nvv, 1.0/nvv)
    xlast = np.copy(x)
    for _ in range(0, n):
        dangling_sum = 0.0
        for v in dangling_nodes:
            dangling_sum += x[v]
        for v in g.vertices:
            xlast[v] = (1 - alpha + alpha * dangling_sum) * (1.0 / nvv)
        for v in g.vertices:
            for u in g.inneighbors(v):
                xlast[v] += (x[u] * a_div_outdegree[u])
        err = 0.0
        for v in g.vertices:
            err += abs(xlast[v] - x[v])
            x[v] = xlast[v]
        if err < (nvv * e):
            return x
    raise Exception("Pagerank did not converge after n iterations.")