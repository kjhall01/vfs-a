"""Short, isolated CPU/float64 VFS throughput campaign; see README.md."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CASES = ('dry', 'moist', 'ocean', 'mixing', 'full', 'full_ale')


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def throughput(dt, seconds):
    return dt / (365 * seconds)


def summarize(samples, steps, dt):
    times = [x['wall_s'] / steps for x in samples]
    median = statistics.median(times)
    return dict(median_step_wall_s=median, min_step_wall_s=min(times),
                max_step_wall_s=max(times), sypd=throughput(dt, median),
                wall_hours_per_simulated_day=24 / (365 * throughput(dt, median)))


def worker(config, output):
    # Imported only after the coordinator sets native thread environment variables.
    import resource
    import torch
    import vfs
    from cases import make_case
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)
    result = dict(config=config, status='running', samples=[],
                  environment=dict(python=sys.version, torch=torch.__version__,
                    platform=platform.platform(), machine=platform.machine(),
                    processor=platform.processor(), logical_cpus=os.cpu_count(),
                    vfs_path=vfs.__file__, torch_threads=torch.get_num_threads(),
                    interop_threads=torch.get_num_interop_threads(),
                    torch_build=torch.__config__.show(), device='cpu', dtype='float64'),
                  source_hashes={str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for root in (HERE, HERE.parent/'ocean_column') for p in sorted(root.glob('*.py'))})
    package = Path(vfs.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(p for p in package.rglob('*') if p.suffix in ('.py','.json')):
        digest.update(str(path.relative_to(package)).encode()); digest.update(path.read_bytes())
    result['installed_python_and_json_sha256'] = digest.hexdigest()
    write(output, result)
    phase = 'setup'
    try:
        with torch.no_grad():
            start = time.perf_counter()
            initial, advance, diagnose = make_case(config)
            before = diagnose(initial)
            result['setup_wall_s'] = time.perf_counter() - start
            phase = 'warmup'
            state = initial
            start = time.perf_counter()
            for i in range(config['warmup']):
                state, _ = advance(state, i)
            result['warmup_wall_s'] = time.perf_counter() - start
            # Each repeat starts from the same state; steps WITHIN a repeat evolve.
            phase = 'measurement'
            for repeat in range(config['repeats']):
                state = initial
                reports = []
                start = time.perf_counter(); cpu = time.process_time()
                for i in range(config['steps']):
                    state, report = advance(state, i)
                    reports.append(report)
                elapsed_cpu = time.process_time() - cpu
                elapsed = time.perf_counter() - start
                # Expensive independent endpoint budgets and report conversion excluded.
                after = diagnose(state)
                from cases import report_metrics
                row = dict(wall_s=elapsed, process_cpu_s=elapsed_cpu,
                           before=before, after=after, reports=report_metrics(reports))
                row['endpoint_changes'] = {key: after[key]-value for key,value in before.items()
                    if isinstance(value,(float,int)) and key in ('mass','volume','salt','passive_inventory','total','energy','boundary_total_energy','tke')}
                if 'total' in before:
                    row['energy_minus_external_inputs_J_kg'] = (
                        after['total']-before['total']-row['reports']['external_energy_J'])/before['initial_mass_kg']
                result['samples'].append(row)
                write(output, result)
            result.update(summarize(result['samples'], config['steps'], config['dt']))
            result['status'] = 'completed'
    except Exception as error:
        import traceback
        result.update(status='failed', failure_phase=phase,
                      error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['peak_process_rss_bytes'] = int(rss if sys.platform == 'darwin' else rss * 1024)
    write(output, result)
    return 0 if result['status'] == 'completed' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preset', choices=('smoke', 'quick', 'scaling'), default='quick')
    p.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    p.add_argument('--sizes', nargs='+', help='paired cell/layer counts, e.g. 1024x64 4096x64')
    p.add_argument('--threads', nargs='+', type=int)
    p.add_argument('--closures', nargs='+', choices=('supported', 'semi', 'fast'), default=['fast'])
    p.add_argument('--dt', type=float, help='actual timestep for every selected case, no stability guarantee')
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--mixing-substeps', type=int, default=4)
    p.add_argument('--warm-start', action='store_true', help='opt-in residual-checked warm ocean dynamics inversions')
    p.add_argument('--remap-every', type=int, default=1)
    p.add_argument('--timeout', type=float, default=1800, help='wall seconds per isolated case, including setup')
    p.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        return worker(json.loads(a.worker.read_text()), a.output)
    sizes = a.sizes or {'smoke':['16x4'], 'quick':['128x16'], 'scaling':['128x32','1024x64','4096x64']}[a.preset]
    try:
        sizes = [tuple(map(int, s.lower().split('x'))) for s in sizes]
        if any(len(s)!=2 or s[0]<16 or s[1]<2 for s in sizes): raise ValueError()
    except ValueError:
        p.error('--sizes needs CELLxLAYER pairs with cells >= 16 and layers >= 2')
    threads = a.threads or ([1] if a.preset == 'smoke' else [1,4,8])
    if min(threads + [a.steps,a.repeats,a.warmup,a.mixing_substeps,a.remap_every]) < 1:
        p.error('thread, step, repeat, warmup and substep counts must be positive')
    if 'full_ale' in a.cases and a.steps < a.remap_every:
        p.error('full_ale needs --steps >= --remap-every to actually time a remap')
    import math
    if not math.isfinite(a.timeout) or a.timeout<=0 or (a.dt is not None and (not math.isfinite(a.dt) or a.dt<=0)):
        p.error('dt and timeout must be finite and positive')
    if a.output.exists(): p.error('output directory already exists; choose a new run directory')
    a.output.mkdir(parents=True)
    rows = []
    try:
        revision = subprocess.check_output(['git','-C',str(HERE),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(['git','-C',str(HERE),'status','--porcelain'],text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        revision,dirty = None,None
    manifest = dict(revision=revision, dirty=dirty, command=sys.argv, cases=rows)
    for cells,layers in sizes:
        for threads_n in threads:
            for case in a.cases:
                for closure in (a.closures if case in ('mixing','full','full_ale') else ['none']):
                    name = f'{case}-{closure}-{cells}x{layers}-t{threads_n}'
                    config = dict(case=case,closure=closure,cells=cells,layers=layers,threads=threads_n,
                        dt=a.dt if a.dt is not None else (1. if a.preset=='smoke' else 60.),
                        steps=a.steps,repeats=a.repeats,warmup=a.warmup,
                        mixing_substeps=a.mixing_substeps,remap_every=a.remap_every,warm_start=a.warm_start)
                    cfg = a.output/(name+'.config.json'); out = a.output/(name+'.json')
                    write(cfg,config)
                    env = dict(os.environ,OMP_NUM_THREADS=str(threads_n),MKL_NUM_THREADS=str(threads_n),
                               OPENBLAS_NUM_THREADS=str(threads_n),VECLIB_MAXIMUM_THREADS=str(threads_n),PYTHONPATH='')
                    print(f'Running {name}, dt={config["dt"]} s',flush=True)
                    with (a.output/(name+'.log')).open('w') as log:
                        try:
                            completed = subprocess.run([sys.executable,'-I',str(__file__),'--worker',str(cfg.resolve()),
                                '--output',str(out.resolve())],env=env,stdout=log,stderr=log,timeout=a.timeout)
                            status = 'completed' if completed.returncode==0 else 'failed'
                        except subprocess.TimeoutExpired:
                            status = 'timeout'
                    result = json.loads(out.read_text()) if out.exists() else {}
                    if status!='completed' or result.get('status')!='completed':
                        result.update(status=status if status!='completed' else 'failed')
                        for key in ('sypd','median_step_wall_s','min_step_wall_s','max_step_wall_s','wall_hours_per_simulated_day'):
                            result.pop(key,None)
                        write(out,result)
                    row = dict(name=name,status=result['status'],config=config)
                    row.update({k:result[k] for k in ('median_step_wall_s','sypd','error') if k in result})
                    rows.append(row); write(a.output/'summary.json',manifest)
                    print(json.dumps(row),flush=True)
    lines = ['# VFS CPU float64 throughput','', '| Case | Status | s/step | SYPD |','|---|---|---:|---:|']
    for r in rows:
        timing = f'{r["median_step_wall_s"]:.4g}' if 'median_step_wall_s' in r else '—'
        rate = f'{r["sypd"]:.4g}' if 'sypd' in r else '—'
        lines.append(f'| {r["name"]} | {r["status"]} | {timing} | {rate} |')
    (a.output/'summary.md').write_text('\n'.join(lines)+'\n')
    return int(any(r['status']!='completed' for r in rows))


if __name__ == '__main__':
    # -I excludes the script directory too; add only our explicit harness path.
    sys.path.insert(0,str(HERE))
    raise SystemExit(main())
