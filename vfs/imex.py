"""Public IMEX import surface.

Active implementations live in :mod:`vfs.integrators.imex`.  This small
facade is retained because ``vfs.imex`` is a documented public import path;
new implementation imports should use the owner package directly.
"""

from .integrators.imex import *  # noqa: F403
from .integrators.imex import __all__
