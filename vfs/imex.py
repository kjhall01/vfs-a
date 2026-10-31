"""Public IMEX import surface.

Active implementations live in :mod:`vfs.atmosphere.imex`.  This small
facade is retained because ``vfs.imex`` is a documented public import path;
new implementation imports should use the owner package directly.
"""

from vfs.atmosphere.imex import *  # noqa: F403
from vfs.atmosphere.imex import __all__
