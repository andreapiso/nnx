import numba
import numpy as np
import heapq

@numba.njit
def standard_distance(s, d):
    return 1.0

@numba.njit
def dijkstra_shortest_paths(g, srcs, allpaths, distf):
    if not srcs:
        raise Exception('Please provide at least one source node')
    nvg = g.nv
    
    dists = np.ones(nvg) * np.infty
    parents = np.full(nvg, -1, dtype=np.int64)
    visited = np.full(nvg, False)
    pathcounts = np.zeros(nvg)
        
    preds = numba.typed.List([numba.typed.List.empty_list(0) for _ in range(nvg)])
    src = srcs[0]
    H = [(0.0, src)] #priority q to be seeded with at least one value
    dists[src] = 0
    visited[src] = True
    pathcounts[src] = 1
    for i in range(1, len(srcs)):
        src = srcs[i]
        dists[src] = 0
        visited[src] = True
        pathcounts[src] = 1
        heapq.heappush(H, (0.0, src))
    
    while H:
        d, u = heapq.heappop(H)
        for v in g.outneighbors(u):
            alt = d + distf(u, v)
            
            if not visited[v]:
                visited[v] = True
                dists[v] = alt
                parents[v] = u
                
                pathcounts[v] += pathcounts[u]
                preds[v] = numba.typed.List([u])
                heapq.heappush(H, (alt, v))
            elif alt < dists[v]: #visited but new path is faster
                dists[v] = alt
                parents[v] = u
                pathcounts[v] = pathcounts[u]
                if allpaths: 
                    preds[v] = numba.typed.List([u])
                heapq.heappush(H, (alt, v))
            elif alt == dists[v]:
                pathcounts[v] += pathcounts[u]
                if allpaths:
                    preds[v].append(u)
    
    for src in srcs:
        pathcounts[src] = 1
        parents[src] = -1
        preds[src] = numba.typed.List.empty_list(0)
            
    return parents, dists, preds