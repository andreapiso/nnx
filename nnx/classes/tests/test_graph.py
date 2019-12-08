from numba import njit
from numba import int64
from numba.tests.support import TestCase, MemoryLeakMixin, unittest

from test_support_base_data import create_complete_graph_edge_list

class TestCreateAddEdgeLength(MemoryLeakMixin, TestCase):
    '''Test graph creation, add edge and length'''
    