import numba
from numba.types import ListType, int64, Set
from numba.experimental import jitclass

simple_g_spec = {
    'ne': int64,
    'fadjlist': ListType(ListType(int64))
}
@jitclass(simple_g_spec)
class _SimpleGraph(object):
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
        return self.fadjlist(v)

    def inneighbors(self, v):
        # return self.badjlist(v)
        return self.neighbors(v)

    def outneighbors(self, v):
        return self.neighbors(v)

    def add_edge(self, s, d):
        nvv = self.nv
        if max(s, d) >= nvv:
            return False
        if d in self.fadjlist[s]:
            return False
        self.fadjlist[s].append(d)
        if d != s:
            self.fadjlist[d].append(s)
        self.ne += 1
        return True

    def rem_edge(self, s, d):
        if max(s, d) >= self.nv:
            return False
        if d not in self.fadjlist[s]:
            return False
        self.fadjlist[s].remove(d)
        self.fadjlist[d].remove(s)
        self.ne -= 1
        return True

    def add_vertex(self):
        #check for overflow?
        self.fadjlist.append(numba.typed.List.empty_list(0))
        
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

@numba.njit
def sg_with_vertices(constructor=0):
    # Create SimpleGraph with n vertices and 0 edges
    fadjlist=numba.typed.List()
    for _ in range(constructor):
        fadjlist.append(numba.typed.List.empty_list(0))
    return _SimpleGraph(0, fadjlist)

@numba.generated_jit(nopython=True)
def SimpleGraph(constructor=0):
    if isinstance(constructor, numba.types.Integer) or isinstance(constructor, numba.types.Omitted):
        return sg_with_vertices
    else:
        raise Exception("Constructor with type {} Not Supported Yet".format(constructor))