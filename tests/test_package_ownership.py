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
