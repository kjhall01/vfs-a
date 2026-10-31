"""Create the separately versioned Gibbs reference; never overwrite evidence.

Requires same-environment captures from the original #18 and #19 wheels.
See verification/ATMOSPHERIC_INTEGRATION.md for reproduction commands.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import torch
import vfs


def load(name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--old-capture', type=Path, required=True)
    parser.add_argument('--original-corrected-capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    common, checker, audit = (load(n) for n in ('run_verification', 'check_baseline', 'audit_fixture_differences'))
    hashes = common.verify_package(args.repo, Path(vfs.__file__).parent, common.NUMERICAL_COMMIT, common.STRUCTURAL_COMMIT)
    source_commit = 'd972f55683bf40b4edf35d1f5193949ff420cb0d'
    source_commit = common.git(args.repo, 'rev-parse', source_commit).decode().strip()
    for name, digest in hashes.items():
        if common.sha256(common.git(args.repo, 'show', f'{source_commit}:{name}')) != digest:
            raise ValueError('Reference source differs: ' + name)
    current = checker.capture(checker.imports())
    before = torch.load(args.old_capture, weights_only=True)
    original = torch.load(args.original_corrected_capture, weights_only=True)
    checker.check(current, original, 'exact')
    historical_path = Path(__file__).resolve().parents[1] / 'tests/fixtures/checkpoint54.pt'
    historical = torch.load(historical_path, weights_only=True)
    # Old same-environment implementation must still satisfy the existing gate.
    checker.check(before, historical, 'portable')
    rows, scalars, structures, species = [], [], [], []
    audit.walk(current, before, '', rows, scalars, structures, species)
    if structures:
        raise ValueError(structures)
    # Explicit scope: initial data, thermodynamic energy definitions and dry
    # dynamics cannot change. Other differences must be in these paired outputs.
    reasons = {
        'rhs': 'Matched Gibbs traces change momentum, entropy and species transport tendencies.',
        'euler': 'Euler uses those tendencies and the authoritative bounded species ledger.',
        'final': 'SSPRK3 propagates the corrected Euler states.',
        'stages': 'Stage limiter decisions respond to the corrected candidate flux.',
        'limiter': 'Candidate species flux and its matched energy correction change together.',
        'final_budgets': 'Diagnostics evaluate the intentionally changed final state.',
        'energy_rate': 'Work contracts the changed tendency; cancellation-level closure is independently checked.',
    }
    for row in rows + scalars:
        parts = row['path'].strip('/').split('/')
        row['reason'] = 'Unchanged in the same environment.'
        if not row['exact']:
            if parts[0] == 'dry' or parts[1] not in reasons:
                raise ValueError('Unexpected changed quantity: ' + row['path'])
            row['reason'] = reasons[parts[1]]
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(current, args.output/'corrected_gibbs_v1.pt')
    # Input states are retained in the fixture. Archive exact geometry too.
    api = checker.imports()
    mesh = api['VoronoiMesh'](n_cells=64, lloyd_iters=0, verbose=False)
    A,B = api['make_hybrid_coefficients'](6,p_top=5000.,sigma_power=1.)
    column = api['CharneyPhillipsColumn'](6,A,B,phi_s=torch.zeros(64),phi_full_mode='direct_u_gl2_adjoint')
    torch.save(dict(mesh=mesh.state_dict(),column=column.state_dict(),initial={k:v['initial'] for k,v in current.items()}),args.output/'inputs.pt')
    common.write_json(args.output/'quantity_comparison.json',dict(tensors=rows,scalars=scalars,species_components=species))
    for name,path in [('old_same_environment.pt',args.old_capture),('original_corrected.pt',args.original_corrected_capture)]:
        shutil.copyfile(path,args.output/name)
    environment = load('record_environment').environment()
    provenance = dict(version='corrected-gibbs-v1',schema='checkpoint54.pt',source_commit=source_commit,
        historical_baseline=common.BASELINE,structural_commit=common.STRUCTURAL_COMMIT,numerical_commit=common.NUMERICAL_COMMIT,
        original_investigation_commit='93b54b3630ba7d17afac87e38e84b8b0080b0d50',original_corrected_commit='24594d55e32d487987fed22c5af0f39b0c9a0426',
        historical_sha256=common.sha256(historical_path.read_bytes()),runtime_sha256=hashes,environment=environment,
        configuration=dict(cells=64,levels=6,p_top_Pa=5000.,dt_s=30.,steps=3,dtype='float64',device='cpu',phi_full_mode='direct_u_gl2_adjoint',cases=list(current)),
        evidence=['verification/GIBBS_FIX_RESULTS.md','verification/ATMOSPHERIC_INTEGRATION.md'],
        files={p.name:common.sha256(p.read_bytes()) for p in args.output.iterdir()})
    for name in ('check_baseline.py','build_corrected_reference.py'):
        shutil.copyfile(Path(__file__).with_name(name),args.output/name)
        provenance['files'][name]=common.sha256((args.output/name).read_bytes())
    common.write_json(args.output/'provenance.json',provenance)


if __name__ == '__main__':
    main()
