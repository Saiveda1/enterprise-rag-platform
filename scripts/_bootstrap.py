"""Import FIRST in every entry script (before numpy/sklearn).

Pins BLAS/OpenMP to a single thread — in this sandbox, thread oversubscription
made a trivial randomized SVD take ~100s; pinned it runs in ~2s and is fully
deterministic. Also forces the Agg matplotlib backend and puts ``src`` on path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
