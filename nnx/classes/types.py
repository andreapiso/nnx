import numba
import numpy as np

def numpy_fadjlist_type():
    return numba.typeof((0, numba.typed.List([np.empty(0, dtype=np.int64)])))

simplegrapharray_const_type = numpy_fadjlist_type()