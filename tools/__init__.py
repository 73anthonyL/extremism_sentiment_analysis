"""Audit toolkit for text classifiers: the two-artifact contract and its tool chain.

The modules in this folder import one another by their bare names, so the
folder is put on the import path when the package is imported. That keeps
`python3 tools/audit.py ...` from a checkout and `esa-audit ...` from an
installed copy running the same code.
"""

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
