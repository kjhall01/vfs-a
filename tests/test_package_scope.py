"""Installed distribution contract for deliberate scope removals."""
import ast
import importlib
import importlib.metadata
import pkgutil
from pathlib import Path

import pytest
import vfs


REMOVED = (
    'atmosphere.dry_dynamics', 'atmosphere.dry_momentum', 'atmosphere.dry_strategies',
    'atmosphere.dry_terms', 'atmosphere.dry_thermo', 'atmosphere.dry_tracers',
    'atmosphere.dry_transport', 'atmosphere.recipes', 'atmosphere.operators',
    'transport', 'forcing', 'split_explicit', 'process', 'external_scm', 'physics',
    'diagnostics', 'core.registry',
    'integrators.imex.ark2', 'integrators.imex.higher_order',
    'integrators.imex.legacy_pressure',
    'ocean.state', 'ocean.column', 'ocean.dynamics', 'ocean.physics', 'ocean_model', 'ocean_diagnostics', 'land', 'ice', 'surface',
    'coupling', 'core.graph', 'geometry.mesh.kernels',
    'atmosphere.pgf_correction', 'transport.neural_reconstruction',
    'transport.shadow_replay', 'transport.reversible_advection',
)


@pytest.mark.parametrize('name', REMOVED)
def test_removed_modules_are_absent_from_installed_distribution(name):
    files = importlib.metadata.files('vfs')
    prefix = 'vfs/' + name.replace('.', '/')
    assert not any(str(f) == prefix + '.py' or str(f).startswith(prefix + '/') for f in files)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('vfs.' + name)


def test_retained_modules_and_lazy_exports_resolve():
    # Exercises atmospheric composition, initialization, I/O, geometry,
    # diagnostics and implicit imports, not just the default import path.
    for info in pkgutil.walk_packages(vfs.__path__, vfs.__name__ + '.'):
        module = importlib.import_module(info.name)
        for name in getattr(module, '_COMPAT_EXPORTS', {}):
            assert getattr(module, name) is not None
    for name in getattr(vfs, '_COMPAT_EXPORTS', {}):
        assert getattr(vfs, name) is not None


def test_removed_exports_and_recipes_are_not_advertised():
    from vfs import atmosphere, geometry
    from vfs.data import topography
    for module, names in (
        (vfs, ('OceanState', 'CoupledState', 'DryHydrostaticDynamics', 'VFSModel', 'HeldSuarezModel')),
        (atmosphere, ('build_dry_dynamics', 'DryOperatorSet', 'DryHydrostaticVirtualTemperatureDynamics')),
        (geometry, ('CellMeshKernelConv',)),
        (geometry.mesh, ('EdgeMeshKernelNetwork',)),
        (topography, ('build_ocean_domain_from_etopo', 'smooth_ocean_depth_on_mesh')),
    ):
        for name in names:
            assert name not in dir(module)
            with pytest.raises(AttributeError):
                getattr(module, name)


def test_retained_source_has_no_imports_from_removed_modules():
    root = Path(vfs.__file__).parent
    for path in root.rglob('*.py'):
        package = 'vfs.' + '.'.join(path.relative_to(root).parts[:-1])
        package = package.rstrip('.')
        for node in ast.walk(ast.parse(path.read_text())):
            targets = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = ('.'.join(package.split('.')[:len(package.split('.')) - node.level + 1])
                        if node.level else '')
                base = '.'.join(p for p in (base, node.module) if p)
                targets = [base] + [base + '.' + a.name for a in node.names]
            for target in targets:
                assert not any(target == 'vfs.' + n or target.startswith('vfs.' + n + '.')
                               for n in REMOVED), (path, node.lineno, target)


def test_only_ark4_and_bounded_explicit_are_public_moist_choices():
    from vfs import imex, atmosphere
    for module, names in ((imex, ("ARK2_TABLEAU", "ARK3_TABLEAU", "DryPEIMEXARK2", "PressureCorrectionCG")),
                          (atmosphere, ("CenteredSSPRK3",))):
        for name in names:
            assert not hasattr(module, name)
    assert imex.ARK4_TABLEAU.order == 4
