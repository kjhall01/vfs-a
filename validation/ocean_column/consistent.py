"""Opt-in research solve with refreshed end-TKE budget acceptance."""
from dataclasses import dataclass
from path_work import PathWorkClosure

@dataclass(frozen=True)
class ConsistentPathClosure(PathWorkClosure):
    refresh_budget: bool = True
    budget_acceleration: bool = True
    max_iterations: int = 320
