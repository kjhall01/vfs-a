"""Compare only completed equal-time, equal-grid path-work endpoints."""
import argparse,json
from pathlib import Path
from analyze_sustained import compare

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data={f.stem:json.loads(f.read_text()) for f in a.input.glob('*.json')}
    data={k:v for k,v in data.items() if 'method' in v}
    rows=[]
    for name,r in sorted(data.items()):
        row={k:r[k] for k in ('method','path_order','scenario','layers','dt_s','status','requested_hours','completed_hours','failure','wall_s','energy_error_J_kg','salt_relative_change','passive_relative_change')}
        row['case']=name;row['iterations']=r['limiter'].get('coupled_iterations',0);rows.append(row)
    comparisons=[]
    for left,right in [('probe','order5'),('probe','cautious'),('path600','path75'),('probe','path75'),('path150','path75')]:
        x,y=data[left],data[right]
        if x['status']==y['status']=='completed' and x['completed_hours']==y['completed_hours'] and x['layers']==y['layers']:
            comparisons.append(dict(left=left,right=right,metrics=compare(x,y)))
    a.output.write_text(json.dumps(dict(cases=rows,comparisons=comparisons,note='Numerical endpoint comparisons, not analytic solutions. Campaign timings overlap other jobs; not a speed benchmark.'),indent=2)+'\n')
if __name__=='__main__':main()
