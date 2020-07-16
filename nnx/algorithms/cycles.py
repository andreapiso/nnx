import numba
import numpy as np

@numba.njit
def _repeated_vertex(v, cycle, n1, n2):
    for k in range(n1, n2+1):
        if cycle[k] == v:
            return True
    return False

@numba.njit
def _simplecycles_lmited_length(graph, n, ceiling, cycles, cycle, i):
    n = min(graph.nv, n)
    if len(cycles) >= ceiling:
        return
    for v in graph.outneighbors(cycle[i]):
        if v == cycle[0]:
            cycles.append(cycle[0:i+1])
        elif (i+1 < n and v > cycle[0] and not _repeated_vertex(v, cycle, 1, i)):
            cycle[i + 1] = v
            #lists are mutable
            _simplecycles_lmited_length(graph, n, ceiling, cycles, cycle, i+1)

@numba.njit
def simplecycles_limited_length(graph, n, ceiling):
    cycles = numba.typed.List.empty_list(np.empty(0, dtype=np.int64))
    if n < 1:
        return cycles
    cycle = np.empty(n, dtype=np.int64)
    for v in graph.vertices:
        cycle[0] = v
        _simplecycles_lmited_length(graph, n, ceiling, cycles, cycle, 0)
        if len(cycles) >= ceiling:
            break
    return cycles