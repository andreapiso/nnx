import numba
from numba.types import ListType, int64, Set
from numba.experimental import jitclass

import nnx.classes.types as nnx_types

simple_g_spec = {
    'ne': int64,
    'fadjlist': ListType(Set(int64))
}
@jitclass(simple_g_spec)
class _SimpleGraphSet(object):
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

    @property
    def edges(self):
        for s in range(self.nv):
            for d in self.neighbors(s):
                if d < s:
                    continue
                yield (s, d)
    
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
    
    def degree(self, v):
        return len(self.neighbors(v))

    def add_edge(self, s, d, add_nodes=False):
        nvv = self.nv
        m = max(s, d)
        if m >= nvv:
            if not add_nodes:
                return False
            self.add_vertices((m - nvv) + 1)
        if d in self.fadjlist[s]:
            return False
        self.fadjlist[s].add(d)
        if d != s:
            self.fadjlist[d].add(s)
        self.ne += 1
        return True

    def add_edges_from(self, edge_iter, add_nodes=False):
        for e in edge_iter:
            self.add_edge(e[0], e[1], add_nodes)

    def rem_edge(self, s, d):
        if max(s, d) >= self.nv:
            return False
        if d not in self.fadjlist[s]:
            return False
        self.fadjlist[s].remove(d)
        if s != d:
            self.fadjlist[d].remove(s)
        self.ne -= 1
        return True

    def add_vertex(self):
        #check for overflow?
        self.fadjlist.append(set(numba.typed.List.empty_list(0)))
        
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

    def is_multigraph(self):
        return False
    
    def has_self_loops(self):
        for v in self.vertices:
            if v in self.neighbors(v):
                return True
        return False

    def self_loop_edges(self):
        self_loops = []
        for v in self.vertices:
            if v in self.neighbors(v):
                self_loops.append((v, v))
        return self_loops

    def number_of_self_loops(self):
        return len(self.self_loop_edges())

    def _get_generator_function(self):
        return SimpleGraphSet

@numba.njit
def sgs_with_vertices(constructor=0):
    # Create SimpleGraph with n vertices and 0 edges
    fadjlist=numba.typed.List()
    for _ in range(constructor):
        fadjlist.append(set(numba.typed.List.empty_list(0)))
    return _SimpleGraphSet(0, fadjlist)

@numba.njit
def _from_grapharray_constructor_tuple(constructor):
    fadjlist = numba.typed.List([set(x) for x in constructor[1]])
    return _SimpleGraphSet(constructor[0], fadjlist)

@numba.generated_jit(nopython=True)
def SimpleGraphSet(constructor=0):
    if isinstance(constructor, numba.types.Integer) or isinstance(constructor, numba.types.Omitted):
        return sgs_with_vertices
    elif constructor == nnx_types.simplegrapharray_const_type:
        return _from_grapharray_constructor_tuple
    else:
        raise Exception("Constructor with type {} Not Supported Yet".format(constructor))