import numba
import numpy as np

@numba.njit
def _standard_neighborfn(g, v):
    return g.outneighbors(v)

@numba.njit
def bfs_traversal(g, ss, neighborfn=_standard_neighborfn):
    n = g.nv
    visited = np.full(n, False)
    parents = np.full(n, -1)
    cur_level = []
    next_level = [0]
    next_level.clear()
    
    for s in ss:
        visited[s] = True
        cur_level.append(s)
        parents[s] = s
    
    while cur_level:
        for v in cur_level:
            for i in neighborfn(g, v):
                if not visited[i]:
                    next_level.append(i)
                    parents[i] = v
                    visited[i] = True
        cur_level.clear()
        cur_level, next_level = next_level, cur_level
        cur_level.sort()
    
    return parents