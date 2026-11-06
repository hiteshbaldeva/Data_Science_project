"""ECG-guided PCG segmentation for HFrEF classification.

This package contains the reusable implementation for the project described in
``README.md`` and ``RESEARCH_DESIGN.md``.  The code is deliberately kept in
small, readable modules so that each signal-processing step can be inspected
and tested independently.
"""

from __future__ import annotations

import os

# On Windows, Anaconda's MKL and PyTorch can each ship a copy of the Intel
# OpenMP runtime.  Loading both aborts the process with "OMP: Error #15".
# This documented workaround must be set before either library is imported;
# it does not change numerical results, but it is a compatibility measure.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
