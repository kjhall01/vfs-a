"""Audit executable ASTs of relocated definitions against merged main.

Imports, annotations and docstrings are excluded; arithmetic, defaults and
statement order remain significant. This supplements installed-wheel captures,
not a replacement for numerical/ownership tests. Explicit output-boundary edits
are listed, never inferred from a failed comparison.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess

BASELINE = '3f806a947e2ff8a5947953137a96af02861d083f'
MOVES = {
    'core/state': ['atmosphere/state'],
    'model': ['atmosphere/dry_model'],
    'geometry/column': ['atmosphere/column'],
    'core/integrators': ['integrators/explicit'],
    'integrators/imex/tableau': ['integrators/tableau'],
    'integrators/imex/solvers': ['atmosphere/imex/solvers', 'integrators/krylov'],
    'geometry/mesh/multilevel': ['geometry/mesh/multilevel', 'atmosphere/imex/transfer'],
    'geometry/regrid': ['geometry/remap'],
    'core/io': ['io', 'atmosphere/io'],
    'data/topography': ['data/elevation', 'atmosphere/topography'],
    'boundary': ['atmosphere/boundary', 'io'],
    **{n: ['atmosphere/'+n] for n in ('earth', 'era5', 'moisture')},
    **{'integrators/imex/'+n: ['atmosphere/imex/'+n]
       for n in ('dry_ark', 'moist_ark', 'conservative_wave', 'moist_wave')},
    **{'initial_conditions/'+n: ['atmosphere/initial_conditions/'+n]
       for n in ('analytic_climate', 'held_suarez', 'equilibrium', 'jablonski_williamson')},
}
BOUNDARIES = {
    'core/state:State.to_latlon': 'Explicit atmospheric adapter; compatibility custom regridders retained.',
    'geometry/regrid:MeshToLatLonKNN.state_to_latlon': 'Split unchanged field calculations into component adapters.',
    'geometry/regrid:_normalise_field_request': 'Defaults owned separately by component output adapters.',
    'core/io:_save_native_state_netcdf': 'Atmosphere assembles fields; shared I/O writes the dataset.',
}


class Executable(ast.NodeTransformer):
    def visit_Import(self, node): return None
    def visit_ImportFrom(self, node): return None
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str): return None
        return self.generic_visit(node)
    def visit_arg(self, node):
        node.annotation = None
        return node
    def visit_FunctionDef(self, node):
        node.returns = None
        return self.generic_visit(node)
    def visit_AnnAssign(self, node):
        return None if node.value is None else ast.Assign(targets=[node.target], value=self.visit(node.value))


def definitions(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef): result[node.name] = node
        if isinstance(node, ast.ClassDef):
            declaration = copy.deepcopy(node)
            declaration.body = [n for n in declaration.body if not isinstance(n, ast.FunctionDef)]
            result[node.name+'.__declaration__'] = declaration
            for child in node.body:
                if isinstance(child, ast.FunctionDef): result[node.name+'.'+child.name] = child
    return result


def run(repo):
    exact, boundary = [], []
    for old, owners in MOVES.items():
        source = subprocess.check_output(['git', '-C', str(repo), 'show', BASELINE+':vfs/'+old+'.py'], text=True)
        targets = {}
        for owner in owners:
            targets.update(definitions((repo/'vfs'/(owner+'.py')).read_text()))
        for name, original in definitions(source).items():
            key = old+':'+name
            if key in BOUNDARIES:
                boundary.append({'path':key, 'reason':BOUNDARIES[key]})
                continue
            current = targets[name]
            normalize = lambda node: ast.dump(Executable().visit(copy.deepcopy(node)), include_attributes=False)
            if normalize(original) != normalize(current): raise AssertionError('Executable AST changed: '+key)
            exact.append(key)
    return {'baseline':BASELINE, 'exact_executable_definitions':exact, 'output_boundaries_checked_by_capture':boundary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.repo)
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    print('Exactly preserved executable definitions:', len(result['exact_executable_definitions']))
