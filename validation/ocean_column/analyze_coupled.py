"""Report completion separately from profile accuracy; never compare unequal times."""
import argparse,json
from pathlib import Path
from analyze_sustained import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();records={}
    for path in sorted(a.input.glob('*.json')):
        r=json.loads(path.read_text())
        if 'requested_hours' in r:records[path.stem]=r
    rows=[]
    for name,r in records.items():
        row={k:r.get(k) for k in ('method','layers','dt_s','requested_hours','completed_hours','status','failure','wall_s','energy_error_J_kg','salt_relative_change','passive_relative_change','accepted_steps','rtol','atol','max_iterations','acceleration','relaxation','source_sha256')}
        row['file']=name+'.json'
        count=r['limiter'].get('coupled_iterations')
        row['mean_iterations']=count/r['accepted_steps'] if count and r['accepted_steps'] else None
        rows.append(row)
    comparisons=[]
    for method in ('coupled','default'):
        ref=records.get(f'short-{method}-11.25')
        if ref is None or ref['status']!='completed':continue
        for dt in (180,90,45):
            name=f'short-{method}-{dt}';r=records.get(name)
            if r and r['status']=='completed' and r['completed_hours']==ref['completed_hours']:
                comparisons.append(dict(case=name,reference=f'short-{method}-11.25',metrics=compare(r,ref)))
    for name in ('robust-600','relaxed-False'):
        left=records.get(name);right=records.get('robust-150')
        if left and right and left['status']==right['status']=='completed' and left['completed_hours']==right['completed_hours']:
            comparisons.append(dict(case=name,reference='robust-150',metrics=compare(left,right)))
    out=dict(reference_note='same-method numerical reference, not analytic truth; interface and dissipation discretizations differ between methods',
        timing_note='concurrent single-thread CPU jobs; indicative cost only, not a whole-model speed benchmark',cases=rows,comparisons=comparisons)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print('cases',len(rows),'comparisons',len(comparisons))

if __name__=='__main__':main()
