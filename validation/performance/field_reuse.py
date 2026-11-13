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
    a=p.parse_args();sys.path.insert(0,str(a.package.resolve()))
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import torch,vfs,hashlib
    from cases import make_case,report_metrics
    from stage_reuse import tensors,counters
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    config=dict(case=a.case,cells=a.cells,layers=a.layers,dt=a.dt,closure='fast' if a.case!='ocean' else 'none',
        mixing_substeps=a.substeps,remap_every=1,warm_start=a.warm,prepared=not a.unprepared)
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
        torch.save(tensors((state,reports)),a.output.with_suffix('.pt'))
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('config','wall_s_per_step','calls','energy_minus_inputs_J_kg')}),flush=True)


if __name__=='__main__':main()
