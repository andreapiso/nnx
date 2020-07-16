import numba
import numpy as np

@numba.njit
def _standard_neighborfn(g, v):
    return g.outneighbors(v)

@numba.njit
def dfs_traversal(g, s, neighborfn=_standard_neighborfn):
    nvv = g.nv
    seen = np.full(nvv, False)
    parents = np.full(nvv, -1)
    S = [s]
    seen[s] = True
    parents[s] = s
    while S:
        v = S[-1]
        u = -1
        for n in neighborfn(g, v):
            
            if not seen[n]:
                u = n
                break
        if u == -1:
            S.pop()
        else:
            seen[u] = True
            S.append(u)
            parents[u] = v
    return parents