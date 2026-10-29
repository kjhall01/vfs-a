"""Measure loaded VFS modules in a fresh process, not timing or memory speedups."""
import json
import sys
import vfs
from vfs.atmosphere import BoundedGibbs, MoistConservativeDynamics
model = BoundedGibbs(MoistConservativeDynamics(), release_as_heat=True)
modules = sorted(name for name in sys.modules if name == 'vfs' or name.startswith('vfs.'))
print(json.dumps({'source': vfs.__file__, 'count': len(modules), 'modules': modules}, indent=2))
