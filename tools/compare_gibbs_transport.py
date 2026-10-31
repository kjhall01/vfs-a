"""Pair #17 C5 archives with proposed-fix runs; never overwrite evidence.

Checks matching configurations, exact initial tensors and runtime source deltas
before comparing all 22 cases. No scientific pass/fail thresholds are imposed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_tree(a,b):
    if isinstance(a,torch.Tensor):
        return (isinstance(b,torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a,b))
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(exact_tree(a[k],b[k]) for k in a)
    return a==b


def discover(roots):
    runs={}
    for root in roots:
        for manifest in root.rglob('manifest.json'):
            record=read(manifest)
            if record['experiment_id']!='C5-H-001':continue
            c=record['configuration']
            key=(c['flow'],c['profile'],c['cells'],c['steps'])
            if key in runs:raise ValueError(f'Duplicate case {key}')
            if record['status']!='completed':raise ValueError(f'Incomplete case {manifest}')
            # Validate archived data, not merely a successful status string.
            for name,expected in record['artifacts_sha256'].items():
                if digest(manifest.parent/name)!=expected:
                    raise ValueError(f'Corrupted artifact {manifest.parent/name}')
            runs[key]=(manifest.parent,record)
    return runs


def summary(folder):
    metrics=read(folder/'metrics.json')
    stages=[json.loads(line) for line in (folder/'stages.jsonl').read_text().splitlines()]
    initial,final=metrics['initial'],metrics['final']
    return dict(metrics=metrics,
        stage_minimum_fraction=np.min([r['minimum_fraction'] for r in stages],axis=0).tolist(),
        stage_maximum_fraction=np.max([r['maximum_fraction'] for r in stages],axis=0).tolist(),
        maximum_stage_constant_composition_error=max(r['constant_composition_error'] for r in stages),
        limited_face_evaluations=sum(r['limited_faces'] for r in stages),
        maximum_donor_substeps=max(r['donor_substeps'] for r in stages),
        maximum_carrier_outflow_fraction=max(r['max_carrier_outflow_fraction'] for r in stages),
        maximum_discrete_divergence_per_s=max(r['max_discrete_velocity_divergence_per_s'] for r in stages),
        exact_euler_ledgers=all(r['exact_water_ledger'] for r in stages),
        final_carrier_relative_change=final['carrier_inventory_kg']/initial['carrier_inventory_kg']-1,
        final_species_relative_change=(np.array(final['species_inventory_kg'])/initial['species_inventory_kg']-1).tolist(),
        stage_count=len(stages),artifact_bytes=sum(p.stat().st_size for p in folder.iterdir()))


def compare(old_roots,new_roots):
    old,new=discover(old_roots),discover(new_roots)
    if old.keys()!=new.keys():raise ValueError('Case sets differ')
    if len(old)!=22:raise ValueError('Expected the same 22 C5 cases')
    rows=[]
    for key in sorted(old):
        a,ma=old[key];b,mb=new[key]
        if ma['configuration']!=mb['configuration']:raise ValueError(f'Configuration changed: {key}')
        if ma['baseline_commit']!=mb['baseline_commit']:raise ValueError('Historical baseline changed')
        if ma.get('numerical_change_commit') is not None:raise ValueError('Old run is not historical')
        if mb.get('numerical_change_commit')!='a7facc1bcdb88f7dc9c10f4fb6059f29b0a9bac7':
            raise ValueError('New run is not the pinned proposed fix')
        if ma['runtime_sha256'].keys()!=mb['runtime_sha256'].keys():
            raise ValueError('Runtime file set changed')
        changed=[p for p in ma['runtime_sha256'] if ma['runtime_sha256'][p]!=mb['runtime_sha256'].get(p)]
        if changed!=['vfs/atmosphere/gibbs_means.py']:raise ValueError(f'Unexpected runtime changes: {changed}')
        for field in ('python','torch','platform'):
            if ma['environment'][field]!=mb['environment'][field]:raise ValueError(f'Environment changed: {field}')
        inputs_a=torch.load(a/'inputs.pt',weights_only=True)
        inputs_b=torch.load(b/'inputs.pt',weights_only=True)
        if not exact_tree(inputs_a,inputs_b):raise ValueError(f'Initial fields/geometry changed: {key}')
        sa,sb=summary(a),summary(b)
        stages_a=[json.loads(s) for s in (a/'stages.jsonl').read_text().splitlines()]
        stages_b=[json.loads(s) for s in (b/'stages.jsonl').read_text().splitlines()]
        if [(r['time_s'],r['dt_s']) for r in stages_a]!=[(r['time_s'],r['dt_s']) for r in stages_b]:
            raise ValueError('Stage times/budgets differ')
        field_differences={}
        for label in ('initial','maximum_deformation','final'):
            xa=torch.load(a/(label+'.pt'),weights_only=True)
            xb=torch.load(b/(label+'.pt'),weights_only=True)
            if xa.keys()!=xb.keys() or any(xa[k].shape!=xb[k].shape or xa[k].dtype!=xb[k].dtype for k in xa):
                raise ValueError(f'State layout changed: {key}/{label}')
            field_differences[label]={name:float((xb[name]-xa[name]).abs().max()) for name in xa}
        rows.append(dict(case=a.name,configuration=ma['configuration'],old=sa,new=sb,
            maximum_state_differences_native_units=field_differences,
            old_manifest_sha256=digest(a/'manifest.json'),new_manifest_sha256=digest(b/'manifest.json'),
            old_artifacts_sha256=ma['artifacts_sha256'],new_artifacts_sha256=mb['artifacts_sha256']))
    return dict(historical_baseline=ma['baseline_commit'],numerical_revision=mb['numerical_change_commit'],
                interpretation='Review evidence, not new fixtures or baseline adoption',cases=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old',type=Path,nargs='+',required=True)
    parser.add_argument('--new',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.old,args.new)
    with args.output.open('x') as output:
        json.dump(result,output,indent=2,allow_nan=False)
        output.write('\n')


if __name__=='__main__':main()
