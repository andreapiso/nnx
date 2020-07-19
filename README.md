# nnx - Numba-compiled network analysis

## NOTE: This library is still in extremely early stage and being actively developed. It is not suitable yet for production use. Use at your own risk!

nnx is a library that leverages the numba compiler (https://numba.pydata.org/) to execute network analysis algorithms at high speed. 

The main strength of nnx compared to libraries that leverage C/Cython to achieve high performance, is that using nnx, custom user code can be sped up as well, as the library makes the numba compiler fully aware of the underlying graph objects. 

### Example Usage - Compute pagerank

```python
import nnx

#produce a binary tree with 65K nodes
g = nnx.balanced_tree(2, 15) 
pagerank_scores = nnx.pagerank(g)
```

### Example Usage - Compute pagerank through a just-in-time compiled function
Since nnx makes numba fully aware of graph objects and algorithms, you can compile your own custom business logic simply by applying the standard numba decorators to your functions.

```python
import numba as nb
import nnx

@nb.njit
def compute_pagerank_within_numba():
    # A graph can be created within the compiled function or passed 
    # as a parameter from python. If created from within numba, performance
    # will be slightly better as the two functions will be inlined.
    g = nnx.balanced_tree(2, 15) 
    pagerank_scores = nnx.pagerank(g)
    return pagerank_scores

pagerank_scores = compute_pagerank_within_numba()
```