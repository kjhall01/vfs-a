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
