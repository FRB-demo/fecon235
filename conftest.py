import sys
import os

# Ensure the parent directory is on sys.path so that 'fecon235' resolves
# to this directory (the package with __init__.py) rather than fecon235.py.
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)
