import numba
from numba.types import ListType, int64
from numba.experimental import jitclass
import numpy as np

import nnx.classes.types as nnx_types

mock_fadj = numba.typed.List()
mock_fadj.append(np.empty(0, dtype=np.int64))
tuple_type = numba.typeof((0, mock_fadj))

simple_g_spec = {
    'ne': int64,
    'fadjlist': numba.typeof(mock_fadj)
}
@jitclass(simple_g_spec)
class _SimpleGraphArray(object):
    # base constructor, providing ne and fadjlist
    def __init__(self, ne, fadjlist):
        self.ne = ne
        self.fadjlist = fadjlist

    @property
    def badj(self):
        return self.fadjlist
    
    @property
    def nv(self):
        return len(self.fadjlist)
    
    @property
    def vertices(self):
        for i in range(self.nv):
            #keep this as generator for now
            yield(i)
    
    def neighbors(self, v):
        return self.fadjlist[v]

    def inneighbors(self, v):
        # return self.badjlist(v)
        return self.neighbors(v)

    def outneighbors(self, v):
        return self.neighbors(v)

    def indegree(self, v):
        return len(self.inneighbors(v))
    
    def outdegree(self, v):
        return len(self.outneighbors(v))

    def add_edge(self, s, d, add_nodes=False):
        nvv = self.nv
        m = max(s, d)
        if m >= nvv:
            if not add_nodes:
                return False
            self.add_vertices((m - nvv) + 1)
        # Get first leg of edge
        search_adjlist = self.fadjlist[s]
        index = np.searchsorted(search_adjlist, d)
        if index < len(search_adjlist) and search_adjlist[index] == d:
            return False # Edge already in graph
        self.fadjlist[s] = np.concatenate((search_adjlist[:index], np.array([d]), search_adjlist[index:]))
        # Handle self loops
        self.ne += 1
        if s == d:
            return True #self loop
        # Complete the other leg of the edge
        search_adjlist = self.fadjlist[d]
        index = np.searchsorted(search_adjlist, s)
        self.fadjlist[d] = np.concatenate((search_adjlist[:index], np.array([s]), search_adjlist[index:]))
        
        return True
    
    def add_edges_from(self, edge_iter, add_nodes=False):
        for e in edge_iter:
            self.add_edge(e[0], e[1], add_nodes)

    def rem_edge(self, s, d):
        nvv = self.nv
        if max(s, d) >= nvv:
            return False
        search_adjlist = self.fadjlist[s]
        index = np.searchsorted(search_adjlist, d)
        if not(index < len(search_adjlist) and search_adjlist[index] == d):
            #edge not in graph
            return False
        self.fadjlist[s] = np.delete(search_adjlist, index)
        self.ne -= 1
        if s != d:
            search_adjlist = self.fadjlist[d]
            index = np.searchsorted(search_adjlist, s)
            self.fadjlist[d] = np.delete(search_adjlist, index)
        return True

    def add_vertex(self):
        #check for overflow?
        self.fadjlist.append(np.empty(0, dtype=np.int64))
        
    def add_vertices(self, n):
        for _ in range(n):
            self.add_vertex()

    def has_edge(self, s, d):
        if max(s, d) >= self.nv:
            return False #edge out of bounds
        if d in self.fadjlist[s]:
            return True
        else:
            return False

    def has_vertex(self, v):
        if v >= self.nv:
            return False
        else:
            return True

    def rem_vertex(self, v):
        n = self.nv - 1
        if v > n:
            return False
        self_loop_n = False

        srcs = self.inneighbors(v).copy() #need copy?
        for s in srcs:
            self.rem_edge(s, v)
        
        if v != n:
            #remove from last vertex
            neigs = self.inneighbors(n).copy()
            for s in neigs:
                if s != n:
                    self.add_edge(s, v)
                else:
                    self_loop_n = True
        
        if self_loop_n:
            self.add_edge(v, v)
        
        self.fadjlist.pop()
        return True

    def is_directed(self):
        return False
    
    def has_self_loops(self):
        if self.nv == 0 or self.ne == 0:
            return False
        for v in self.vertices:
            index = np.searchsorted(self.neighbors(v), v)
            if index <= len(self.neighbors(v)) and self.neighbors(v)[index] == v:
                return True
        return False

    def self_loop_edges(self):
        self_loops = []
        if self.nv == 0 or self.ne == 0:
            return self_loops
        for v in self.vertices:
            index = np.searchsorted(self.neighbors(v), v)
            if index <= len(self.neighbors(v)) and self.neighbors(v)[index] == v:
                self_loops.append((v, v))
        return self_loops

    def number_of_self_loops(self):
        return len(self.self_loop_edges())

@numba.njit
def sg_with_vertices(constructor=0):
    # Create SimpleGraph with n vertices and 0 edges
    fadjlist=numba.typed.List()
    # fadjlist=numba.typed.List.empty_list(np.empty(0, dtype=np.int64))
    for _ in range(constructor):
        fadjlist.append(np.empty(0, dtype=np.int64))
    return _SimpleGraphArray(0, fadjlist)

@numba.njit
def _from_constructor_tuple(constructor):
    return _SimpleGraphArray(constructor[0], constructor[1])

@numba.generated_jit(nopython=True)
def SimpleGraphArray(constructor=0):
    if isinstance(constructor, numba.types.Integer) or isinstance(constructor, numba.types.Omitted):
        return sg_with_vertices
    # elif isinstance(constructor, (tuple_type)):
    #     return sg_from_numpy_fadjlist
    elif constructor == nnx_types.simplegrapharray_const_type:
        return _from_constructor_tuple
    else:
        raise Exception("Constructor with type {} Not Supported Yet".format(constructor))

