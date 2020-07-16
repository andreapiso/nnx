import numba
import nnx
import numpy as np

@numba.njit
def _connected_components(g):
    nvg = g.nv
    label = np.full(nvg, -1)
    
    for u in g.vertices:
        if label[u] != -1:
            continue
        label[u] = u
        Q = [u]
        while len(Q) > 0:
            src = Q.pop()
            for vertex in g.neighbors(src):
                if label[vertex] == -1:
                    Q.append(vertex)
                label[vertex] = u
    return label

@numba.njit
def _components(labels):
    d = numba.typed.Dict()
    c = [[0]]
    c.clear()
    i = 0
    for v in range(len(labels)):
        l = labels[v]
        if l not in d:
            d[l] = i
        index = d[l]
        if len(c) > index:
            c[index].append(v)
        else:
            c.append([v])
            i += 1
            
    return c, d

@numba.njit
def connected_components(g):
    label = _connected_components(g)
    c, d = _components(label)
    return c