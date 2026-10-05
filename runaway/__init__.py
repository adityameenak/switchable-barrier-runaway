"""1D thermal runaway propagation in a Li-ion cell stack with switchable barriers.

All physical parameters are ILLUSTRATIVE; see ``runaway.params``.
"""

from .params import (
    DESIGNS,
    BarrierParams,
    CellParams,
    KineticsParams,
    SimParams,
    StackParams,
)
from .solver import simulate_runaway, solve_normal_operation

__all__ = [
    "DESIGNS",
    "BarrierParams",
    "CellParams",
    "KineticsParams",
    "SimParams",
    "StackParams",
    "simulate_runaway",
    "solve_normal_operation",
]
__version__ = "0.1.0"
