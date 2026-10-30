"""Installed distribution contract for deliberate scope removals."""
import ast
import importlib
import importlib.metadata
import pkgutil
from pathlib import Path

import pytest
import vfs


REMOVED = (
    'ocean', 'ocean_model', 'ocean_diagnostics', 'land', 'ice', 'surface',
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
    for name in vfs._COMPAT_EXPORTS:
        assert getattr(vfs, name) is not None


def test_removed_exports_and_recipes_are_not_advertised():
    from vfs import atmosphere, geometry
    from vfs.atmosphere import recipes, dry_strategies
    from vfs.data import topography
    from vfs.core.registry import get_component
    from vfs.model import VFSModel, _build_component
    for module, names in (
        (vfs, ('OceanState', 'CoupledState', 'AirSeaMomentumCoupler')),
        (atmosphere, ('PGFPotentialCorrectionMLP', 'LearnedAdaptiveDivergenceDamping',
                      'DryHydrostaticLearnedSoftMUSCLMthetaTransportDynamics')),
        (recipes, ('MthetaMUSCLActivatorMLP',)),
        (dry_strategies, ('LearnedAdaptiveDivergenceDamping',)),
        (geometry, ('CellMeshKernelConv',)),
        (geometry.mesh, ('EdgeMeshKernelNetwork',)),
        (topography, ('build_ocean_domain_from_etopo', 'smooth_ocean_depth_on_mesh')),
    ):
        for name in names:
            assert name not in dir(module)
            with pytest.raises(AttributeError):
                getattr(module, name)
    assert get_component('air_sea_momentum_coupler') is None
    assert 'learned_soft_muscl_mtheta_transport' not in recipes.dry_operator_sets()
    with pytest.raises(ValueError, match='unknown'):
        VFSModel.from_config({'model': 'forced_ocean'})
    with pytest.raises(ValueError, match='unknown'):
        _build_component('surface_system', name='surface', mesh=None, column=None)


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
