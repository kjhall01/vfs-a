"""One isolated wheel-based benchmark sample; outputs tensors locally for comparison.

--package points to a wheel unpacked outside the checkout. Alternate independent
parent/candidate invocations; do not run concurrent timing jobs. Tensor captures
are local verification intermediates, never upload files above 128 KiB.
"""
import argparse,json,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=['ocean','full','full_ale'],default='ocean')
    p.add_argument('--cells',type=int,default=128);p.add_argument('--layers',type=int,default=16)
    p.add_argument('--dt',type=float,default=60.);p.add_argument('--steps',type=int,default=2)
    p.add_argument('--substeps',type=int,default=4);p.add_argument('--warm',action='store_true')
    p.add_argument('--unprepared',action='store_true')
    p.add_argument('--energy-quadrature',type=int,default=4)
    p.add_argument('--compile-thermo',action='store_true',help='research JMD95 kernel compilation; no whole-model guarantee')
    p.add_argument('--compile-gibbs',action='store_true',help='research surface Gibbs/caloric kernel fusion; guards remain eager')
    a=p.parse_args();sys.path.insert(0,str(a.package.resolve()))
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import torch,vfs,hashlib
    from cases import make_case,report_metrics
    from stage_reuse import tensors,counters
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    config=dict(case=a.case,cells=a.cells,layers=a.layers,dt=a.dt,closure='fast' if a.case!='ocean' else 'none',
        mixing_substeps=a.substeps,remap_every=1,warm_start=a.warm,prepared=not a.unprepared,energy_quadrature=a.energy_quadrature)
    original_kernel = None
    original_gibbs = None
    if a.compile_thermo and not a.compile_gibbs:
        import vfs.ocean.jmd95_energy as jmd
        original_kernel = jmd.jmd95_partials
        jmd.jmd95_partials = torch.compile(original_kernel,fullgraph=True)
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ocean_column'))
        import fast_thermo
        fast_thermo.jmd95_partials = jmd.jmd95_partials
        fast_thermo.temperature_partial = torch.compile(fast_thermo.temperature_partial,fullgraph=True)
    if a.compile_gibbs and not a.compile_thermo:
        from vfs.ocean.stratified.thermodynamics import SurfaceGibbs
        original_gibbs = SurfaceGibbs._values
        SurfaceGibbs._values = torch.compile(original_gibbs,fullgraph=True,dynamic=True)
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ocean_column'))
        import fast_thermo
        fast_thermo.Caloric.evaluate = torch.compile(fast_thermo.Caloric.evaluate,fullgraph=True,dynamic=True)
    config['compile_gibbs'] = a.compile_gibbs
    config['compile_thermo'] = a.compile_thermo
    with torch.no_grad():
        initial,advance,diagnose=make_case(config)
        def run():
            state=initial;reports=[]
            for i in range(a.steps):state,report=advance(state,i);reports.append(report)
            return state,reports
        run()
        start=time.perf_counter();state,reports=run();elapsed=time.perf_counter()-start
        before=diagnose(initial);after=diagnose(state);ledger=report_metrics(reports)
        with counters() as count:run()
        result=dict(config=config,steps=a.steps,wall_s_per_step=elapsed/a.steps,calls=count,
            before=before,after=after,ledger=ledger,
            energy_minus_inputs_J_kg=(after['total']-before['total']-ledger['external_energy_J'])/before['initial_mass_kg'],
            torch=torch.__version__,package=vfs.__file__,
            source_hashes={str(x.relative_to(Path(vfs.__file__).parent)):hashlib.sha256(x.read_bytes()).hexdigest()
                for x in sorted(Path(vfs.__file__).parent.joinpath('ocean').rglob('*.py'))})
        if original_kernel is not None:
            jmd.jmd95_partials = original_kernel
        if original_gibbs is not None:
            SurfaceGibbs._values = original_gibbs
        from vfs.ocean.stratified import StratifiedOcean,TKEOcean
        reference = TKEOcean(StratifiedOcean(initial.ocean.mesh,initial.ocean.column,quadrature_order=8))
        reference_change = reference.budgets(state)['total']-reference.budgets(initial)['total']-ledger['external_energy_J']
        result['order8_energy_minus_inputs_J_kg'] = float(reference_change)/before['initial_mass_kg']
        torch.save(tensors((state,reports)),a.output.with_suffix('.pt'))
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('config','wall_s_per_step','calls','energy_minus_inputs_J_kg')}),flush=True)


if __name__=='__main__':main()
