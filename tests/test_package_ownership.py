"""Ownership migration: aliases identify the same implementation and tensors."""
import importlib


def test_atmospheric_public_aliases_have_one_owner():
    from vfs import State, DryAtmosphere, CharneyPhillipsColumn
    for cls, owner, old in (
        (State, 'vfs.atmosphere.state', 'vfs.core.state'),
        (DryAtmosphere, 'vfs.atmosphere.dry_model', 'vfs.model'),
        (CharneyPhillipsColumn, 'vfs.atmosphere.column', 'vfs.geometry.column'),
    ):
        assert cls.__module__ == owner
        assert getattr(importlib.import_module(old), cls.__name__) is cls
    from vfs.geometry.column import make_hybrid_coefficients as old
    from vfs.atmosphere.column import make_hybrid_coefficients
    assert old is make_hybrid_coefficients


def test_shared_integration_has_no_component_imports():
    import ast
    import inspect
    for name in ('explicit', 'tableau', 'krylov'):
        module = importlib.import_module('vfs.integrators.' + name)
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, ast.ImportFrom):
                assert not any(x in (node.module or '') for x in ('atmosphere', 'ocean', 'core.state'))
    from vfs.imex import DryPEIMEXARK
    from vfs.atmosphere.imex import DryPEIMEXARK as owner
    from vfs.integrators.imex.solvers import _gmres as old
    from vfs.integrators.krylov import _gmres
    assert DryPEIMEXARK is owner
    assert old is _gmres


def test_shared_geometry_data_and_io_have_no_component_dependencies():
    import ast
    import inspect
    for name in ('vfs.geometry.remap', 'vfs.geometry.mesh.multilevel',
                 'vfs.data.elevation', 'vfs.io'):
        module = importlib.import_module(name)
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, ast.ImportFrom):
                assert not any(x in (node.module or '') for x in ('atmosphere', 'ocean', 'core.state'))
    from vfs.geometry.remap import MeshToLatLonKNN
    assert not hasattr(MeshToLatLonKNN, 'state_to_latlon')


def test_explicit_atmospheric_output_adapter_preserves_fields_and_gradients():
    import torch
    from vfs.field import Field
    from vfs.geometry.remap import LatLonGrid, MeshToLatLonKNN
    from vfs.geometry.regrid import MeshToLatLonKNN as LegacyRemapper
    from vfs.atmosphere.regrid import state_to_latlon
    from test_conservative_imex import setup
    _, state, _ = setup()
    state.u_e.requires_grad_()
    grid = LatLonGrid.regular(nlat=8, nlon=16)
    shared = MeshToLatLonKNN.from_mesh(state.mesh, grid)
    legacy = LegacyRemapper.from_mesh(state.mesh, grid)
    class CustomRemapper(LegacyRemapper):
        def state_to_latlon(self, supplied, *, fields='default'):
            return {'same_state': supplied is state, 'request': fields}
    custom = CustomRemapper.from_mesh(state.mesh, grid)
    assert state.to_latlon(custom, fields='all') == {'same_state': True, 'request': 'all'}
    before = tuple(getattr(state, k).clone() for k in ('u_e', 'ps', 'mtheta_i'))
    explicit = state_to_latlon(shared, state, fields='all')
    for alternative in (state.to_latlon(shared, fields='all'), legacy.state_to_latlon(state, fields='all')):
        assert alternative.keys() == explicit.keys()
        for name, value in explicit.items():
            if isinstance(value, Field):
                torch.testing.assert_close(value.values, alternative[name].values, rtol=0, atol=0)
                assert value.location == alternative[name].location
                assert value.vertical_location == alternative[name].vertical_location
                assert value.units == alternative[name].units
    first = torch.autograd.grad(explicit['wind_speed'].values.sum(), state.u_e)[0]
    second = torch.autograd.grad(alternative['wind_speed'].values.sum(), state.u_e)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for name, original in zip(('u_e', 'ps', 'mtheta_i'), before):
        torch.testing.assert_close(getattr(state, name), original, rtol=0, atol=0)


def test_new_and_compatibility_initialization_imports_in_fresh_processes():
    import subprocess
    import sys
    for first in ('vfs.atmosphere.topography', 'vfs.atmosphere.era5',
                  'vfs.data.elevation', 'vfs.data.topography'):
        subprocess.run([sys.executable, '-I', '-c', f'''
import {first}
from vfs.earth import earth_like_dry_initial_state as old
from vfs.atmosphere.earth import earth_like_dry_initial_state
assert old is earth_like_dry_initial_state
from vfs.data import build_topographic_column
from vfs.atmosphere.topography import build_topographic_column as owner
assert owner is build_topographic_column
'''], check=True, capture_output=True, text=True)
