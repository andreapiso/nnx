import numba
import nnx
import numpy as np

default_graphtype = nnx.SimpleGraphArray

@numba.njit
def empty_graph(n=0, create_using=default_graphtype):
    return create_using(n)

############## r-ary balanced tree #################

@numba.njit
def _optimised_tree_edges(n, r):
    current_parent = 0
    current_count = 0
    for i in range(1, n):
        if current_count == r:
            current_parent += 1
            current_count = 0
        current_count += 1
        yield current_parent, i

@numba.njit
def _tree_edges(n, r):
    if n == 0:
        return
    # helper function for trees
    # yields edges in rooted tree at 0 with n nodes and branching ratio r
    nodes = iter(range(n))
    parents = [next(nodes)]  # stack of max length r
    while parents:
        break_point = False
        source = parents.pop(0)
        for i in range(r):
            if break_point:
                break
            try:
                target = next(nodes)
                parents.append(target)
                yield source, target
            except:
                # Workaround on numba bug where break-point cannot be in try-except block
                break_point = True

@numba.njit
def full_rary_tree(r, n, create_using=default_graphtype):
    G = empty_graph(n, create_using)
    G.add_edges_from(_optimised_tree_edges(n, r))
    return G

@numba.njit
def balanced_tree(r, h, create_using=default_graphtype):
    """
    Notes
    -----
    This is the rooted tree where all leaves are at distance `h` from
    the root. The root has degree `r` and all other internal nodes
    have degree `r + 1`.

    Node labels are integers, starting from zero.

    A balanced tree is also known as a *complete r-ary tree*.

    """
    # The number of nodes in the balanced tree is `1 + r + ... + r^h`,
    # which is computed by using the closed-form formula for a geometric
    # sum with ratio `r`. In the special case that `r` is 1, the number
    # of nodes is simply `h + 1` (since the tree is actually a path
    # graph).
    if r == 1:
        n = h + 1
    else:
        # This must be an integer if both `r` and `h` are integers. If
        # they are not, we force integer division anyway.
        n = (1 - r ** (h + 1)) // (1 - r)
    return full_rary_tree(r, n, create_using=create_using)

# @numba.njit
# def _pair_permutations(l):
#     n = len(l)
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 continue
#             yield i, j

# @numba.njit
# def _pair_combinations(l):
#     n = len(l)
#     for i in range(n):
#         for j in range(i + 1, n):
#             yield i, j

@numba.njit
def complete_graph(n, create_using=default_graphtype):
    """ Return the complete graph `K_n` with n nodes.

    Examples
    --------
    >>> G = nnx.complete_graph(9)
    >>> len(G)
    9
    >>> G.size()
    36
    """
    #TODO: complete graph from iterable. Maybe using @generated_jit?
    if n <= 0:
        return create_using(0)
    ne = int(n * (n - 1) // 2)
    fadjlist = numba.typed.List()
    for u in range(n):
        listu = np.empty(n-1, dtype=np.int64)
        listu[0:u] = np.arange(0,u)
        listu[u:n] = np.arange(u+1, n)
        fadjlist.append(listu)
    return create_using((ne, fadjlist))

@numba.njit
def complete_bipartite_graph(n1, n2, create_using=default_graphtype):
    """
    complete_bipartite_graph(n1, n2, create_using)

    Create an undirected [complete bipartite graph](https://en.wikipedia.org/wiki/Complete_bipartite_graph)
    with `n1 + n2` vertices.

    Composed of two partitions with n1 nodes in the first and n2 nodes in the second. Each node in the first is connected to each node in the second.
    """
    if n1 < 0 or n2 < 0:
        return create_using()
    
    n = n1 + n2
    ne = n1 * n2
    
    range1 = np.arange(0, n1)
    range2 = np.arange(n1, n)
    
    fadjlist = numba.typed.List()
    
    for _ in range(0, n1):
        fadjlist.append(np.copy(range2))
    for _ in range(n1, n):
        fadjlist.append(np.copy(range1))
    
    return create_using((ne, fadjlist))

@numba.njit
def star_graph(n, create_using):
    """
    Create an undirected [star graph](https://en.wikipedia.org/wiki/Star_(graph_theory))
    with `n+1` vertices: One center node, connected to n outer nodes.
    """
    if n <= 0:
        return create_using()
    ne = n
    fadjlist = numba.typed.List([np.arange(1, n+1)])
    for _ in range(1, n+1):
        fadjlist.append(np.array([0]))
    return create_using((ne, fadjlist))

@numba.njit
def path_graph(n, create_using):
    """
    Create a [path graph](https://en.wikipedia.org/wiki/Path_graph)
    with `n` vertices.
    """
    if n <= 1:
        return create_using(n)
    
    ne = n - 1
    fadjlist = numba.typed.List()
    fadjlist.append(np.array([1]))
    for i in range(1, n-1):
        fadjlist.append(np.array([i - 1, i + 1]))
    fadjlist.append(np.array([n-2]))
    
    return create_using((ne, fadjlist))

@numba.njit
def circulant_graph(n, offsets, create_using=default_graphtype):
    """Generates the circulant graph $Ci_n(x_1, x_2, ..., x_m)$ with $n$ vertices.

    Returns
    -------
    The graph $Ci_n(x_1, ..., x_m)$ consisting of $n$ vertices $0, ..., n-1$ such
    that the vertex with label $i$ is connected to the vertices labelled $(i + x)$
    and $(i - x)$, for all $x$ in $x_1$ up to $x_m$, with the indices taken modulo $n$.

    """
    G = create_using(n)
    for i in range(n):
        for j in offsets:
            G.add_edge(i, (i - j) % n)
            G.add_edge(i, (i + j) % n)
    return G

# @numba.njit()
# def pairwise(l, cyclic=False):
#     n = len(l)
#     edges = numba.typed.List()
#     for i in range(n-1):
#         edges.append((l[i], l[i+1]))
#     if cyclic:
#         edges.append((l[n-1], l[0]))
#     return edges

