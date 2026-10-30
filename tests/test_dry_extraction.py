"""Retained dry regression after removing the historical method-binding table."""
import ast
import importlib.util
import inspect
from pathlib import Path
import torch
from vfs.atmosphere.conservative import DryHydrostaticConservativeDynamics
from vfs.atmosphere.dry_energy import DryEnergy

ROOT = Path(__file__).resolve().parents[1]


def test_equations_are_declared_methods_not_external_binding_tables():
    for owner in (DryHydrostaticConservativeDynamics, DryEnergy):
        tree = ast.parse(inspect.getsource(owner))
        for node in tree.body[0].body:
            if isinstance(node, ast.Assign):
                assert not isinstance(node.value, ast.Attribute)


def test_expanded_retained_dry_baseline(fixture_mode):
    path = ROOT / 'tools/check_dry_helpers.py'
    spec = importlib.util.spec_from_file_location('dry_helpers', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.check(mod.capture(), torch.load(ROOT/'tests/fixtures/dry_helpers_pass2.pt', weights_only=True), fixture_mode)
    assert result['tensor_comparisons'] == 93
