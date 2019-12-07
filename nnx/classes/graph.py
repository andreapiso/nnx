'''
Base classes for undirected graphs representation.

BaseGraph is an adjacency list representation of a graph where every node ID is
an integer and the id of a node represents its position in the adj list of the
graph.

Graph (TODO) is a symbolic graph where the ID of a node can be of any type
and a dictionary maps the ID with its BaseGraph physical representation.
'''

import numba
from numba.types import ListType, int64

'''
Attributes for the base graph class. New attributes can be added by
modifying the spec and extending the base class with the new spec.
'''
base_graph_spec = {
    '_E': int64,
    '_node': ListType(ListType(int64))
}
@numba.jitclass(base_graph_spec)
class BaseGraph(object):
    
    def __init__(self):
        self._E = 0
        adj = numba.typed.List()
        l = numba.typed.List.empty_list(0)
        # FIXME due to a numba quirk, cannot instantiate a List of type 
        # List[List[int64]] without creating one instance of the internal
        # lists as well. This means empty graphs actually have a "fake" 
        # node without edges. Should add another list to keep track of 
        # valid positions in the adj list. This would be useful in deleting
        # nodes as well without invalidating the position of all the other
        # nodes.
        adj.append(l)
        self._node = adj

    # TODO: add additional non-fundamental properties such as "name"

    def __str__(self):
        s = ''
        for v in range(len(self)):
            for u in self._node[v]:
                s = s + str(v) + '->' + str(u) + '\n'
        return s

    def __iter__(self):
        '''
        Iterates over graph nodes. Every element is list of neighbours
        of the node.
        '''
        return iter(self._node)

    def __contains__(self,n):
        '''
        Return True if n is a node. 
        FIXME: like everything else, this function assumes no node is 
        ever deleted.
        On the other hand, with that assumption, we can just simply 
        check the length of the nodes list which is very fast.
        '''
        return len(self._node) > n

    def __len__(self):
        '''
        Returns number of nodes in graph. 
        FIXME: assumes no deletion, empty graph will give wrong result
        '''
        return len(self._node)

    def __getitem__(self, n):
        return self._node[n]

    def add_edge(self, u, v):
        '''
        Adds edge to a graph. If nodes do not exist, adds them.
        TODO: right now edges have no attributes
        '''
        while len(self) <= max(u, v):
            self._node.append(numba.typed.List.empy_list(0))
        # TODO: should check if v is already in node[u] or allow for
        # parallel edges?
        self._node[u].append(v)
        self._node[v].append(u)
        self._E += 1

    def remove_edge(self, u, v):
        self._node[u].remove(v)
        self._node[v].remove(u)

    def add_edges_from(self, ebunch_to_add):
        '''
        Adds edges from an edge container. Edges needs to be 
        represented in the container as 2-tuples of ints.
        '''
        for u, v in ebunch_to_add:
            self.add_edge(u, v)

    def copy(self):
        copy_g = BaseGraph()
        while len(copy_g) < len(self):
            copy_g._node.append(numba.typed.List.empty_list(0))
        for v in range(len(self)):
            for w in self._node[v]:
                copy_g._node[v].append(w)
        copy_g.E = self.E
        return copy_g

    
