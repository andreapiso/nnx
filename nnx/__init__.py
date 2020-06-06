import sys
if sys.version_info[:2] < (3, 6):
    m = "Python 3.6 or later is required for nnx (%d.%d detected)."
    raise ImportError(m % sys.version_info[:2])
del sys

import nnx.classes
from nnx.classes.simplegraphset import *
from nnx.classes.simplegraph import *
import nnx.jit