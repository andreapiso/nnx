import numba
import numpy as np

@numba.njit
def induced_subgraph(g, vlist):
    vlen = len(vlist)
    h = g._get_generator_function()(vlen)
    vset = set(vlist)
    if len(vset) != vlen:
        raise Exception("Vertices in subgraph list must be unique")
    newid = dict()
#     vmap = np.full(len(vlist), -1) #comment out vmap as it seems it's just vlist...
    # NOTE: vmap might be important if passing a list of edges instead. might be good to keep it in that case
    for i, v in enumerate(vlist):
        newid[v] = i
#         vmap[i] = v
    
    for s in vlist:
        for d in g.outneighbors(s):
            if d in vset and g.has_edge(s, d):
                h.add_edge(newid[s], newid[d])
    
    return h #, vmap #commented out right now, can get through vlist?

@numba.njit
def intersect(g, h):
    if g.is_multigraph() != h.is_multigraph():
        raise Exception("Intersection requires both operands to be graphs or multigraphs")
    gnv = g.nv
    hnv = h.nv
    
    r = g._get_generator_function()(min(gnv, hnv))
    for s, d in g.edges:
        if h.has_edge(s, d):
            r.add_edge(s, d)
    return r