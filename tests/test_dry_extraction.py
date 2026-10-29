"""Structural and numerical regression checks for removing dry mixins."""
import ast
import hashlib
import inspect
import importlib.util
import json
from pathlib import Path
import torch
from vfs.atmosphere.dry_dynamics import DryHydrostaticDynamics

ROOT=Path(__file__).resolve().parents[1]


def test_no_dry_mixin_inheritance():
    assert DryHydrostaticDynamics.__bases__ == (torch.nn.Module,)


def normalized_ast(node):
    """Ignore only docstring indentation introduced by removing the class."""
    if ast.get_docstring(node, clean=False) is not None:
        node.body[0].value.value = inspect.cleandoc(node.body[0].value.value)
    return ast.dump(node, include_attributes=False)


def test_every_extracted_function_has_identical_syntax():
    import vfs.atmosphere
    directory=Path(vfs.atmosphere.__file__).parent
    expected=json.loads((ROOT/'tests/fixtures/dry_method_ast_pass2.json').read_text())
    for name,checksum in expected.items():
        module,function=name.split('.')
        tree=ast.parse((directory/(module+'.py')).read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
        assert hashlib.sha256(normalized_ast(node).encode()).hexdigest()==checksum,name
        owner=getattr(__import__('vfs.atmosphere.'+module,fromlist=[function]),function)
        assert getattr(DryHydrostaticDynamics,function) is owner


def test_expanded_dry_baseline():
    path=ROOT/'tools/check_dry_helpers.py'
    spec=importlib.util.spec_from_file_location('dry_helpers',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    count=mod.compare(mod.capture(),torch.load(ROOT/'tests/fixtures/dry_helpers_pass2.pt',weights_only=True))
    assert count>100
