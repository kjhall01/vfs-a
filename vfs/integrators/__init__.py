"""Time-integration implementations grouped by numerical method.

The low-level explicit Runge--Kutta helpers remain in ``vfs.core`` because
they depend only on the generic state protocol.  Domain-aware IMEX methods
live in :mod:`vfs.integrators.imex` instead.
"""

__all__: list[str] = []
