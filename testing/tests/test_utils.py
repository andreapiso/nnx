# Tests the basic utils
import nnx
import numba
from testing import assert_graphs_equal

class _GenericTest:
    @classmethod
    def _test_equal(cls, a, b):
        cls._assert_func(a, b)

    @classmethod
    def _test_not_equal(cls, a, b):
        try:
            cls._assert_func(a, b)
            passed = True
        except AssertionError:
            pass
        else:
            raise AssertionError("a and b are found equal but are not")

class TestGraphsEqual(_GenericTest):
    _assert_func = assert_graphs_equal


    def test_graphs_equal(self):
        g = nnx.from_edge_list(numba.typed.List([(0,1), (1,2), (2,3)]), create_using=nnx.SimpleGraphArray)
        h = nnx.path_graph(4, create_using=nnx.SimpleGraphArray)
        self._test_equal(g, h)

    def test_graphs_not_equal(self):
        g = nnx.from_edge_list(numba.typed.List([(0,1), (1,2)]), create_using=nnx.SimpleGraphArray)
        h = nnx.path_graph(4, create_using=nnx.SimpleGraphArray)
        self._test_not_equal(g, h)