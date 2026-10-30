"""Check import boundaries in fresh processes, independent of test order."""
import subprocess
import sys
import json


def run(code):
    return subprocess.run([sys.executable, '-c', code], check=True, text=True, capture_output=True).stdout


def test_default_model_does_not_load_optional_systems():
    names = json.loads(run('''
import sys, json
import vfs
model = vfs.BoundedGibbs(vfs.MoistConservativeDynamics(), release_as_heat=True)
print(json.dumps([n for n in sys.modules if n.startswith("vfs.")]))
'''))
    excluded = ('vfs.ocean', 'vfs.land', 'vfs.ice', 'vfs.coupling',
                'vfs.core.graph', 'vfs.transport', 'vfs.atmosphere.recipes',
                'vfs.atmosphere.pgf_correction', 'vfs.geometry.mesh.kernels',
                'vfs.geometry.mesh.multilevel', 'vfs.geometry.regrid')
    assert not [n for n in names if n.startswith(excluded)]


def test_explicit_compatibility_imports_and_unknown_names():
    run('''
import vfs
from vfs.atmosphere import build_dry_dynamics
from vfs.geometry.mesh import VoronoiMeshHierarchy
from vfs.geometry import LatLonGrid
from vfs.imex import DryPEIMEXARK, MoistPEIMEXARK
assert build_dry_dynamics("conservative_diagonal").recipe_name == "conservative_diagonal"
assert vfs.LatLonGrid is LatLonGrid
try:
    vfs.this_name_does_not_exist
except AttributeError:
    pass
else:
    raise AssertionError("unknown exports must fail")
''')


def test_removed_learned_pressure_option_fails_explicitly():
    run('''
from vfs import DryHydrostaticDynamics
try:
    DryHydrostaticDynamics(pgf_correction_mode='learned_potential')
except ValueError as error:
    assert 'removed' in str(error)
else:
    raise AssertionError('removed learned PGF mode must fail')
''')
