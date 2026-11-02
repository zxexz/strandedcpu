"""Compile small Python and C programs, then check their numeric results.

The interpreter is the default execution backend. CUDA and the analytical
transformer provide additional execution paths for the same compiled programs.
"""

from strandedcpu.api import run_function, score_options, score_options_batch
from strandedcpu.compiler import ExecutionFault
from strandedcpu.python import score_python, score_python_batch

__all__ = [
    "ExecutionFault",
    "run_function",
    "score_options",
    "score_options_batch",
    "score_python",
    "score_python_batch",
]
