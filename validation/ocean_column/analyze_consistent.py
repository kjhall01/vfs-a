"""Compare completed research runs; never compare failed prefixes as endpoints."""
import argparse,json
from pathlib import Path
from analyze_sustained import compare

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 data={f.stem:json.loads(f.read_text()) for f in a.input.glob('*.json')}
 data={n:r for n,r in data.items() if 'solver' in r}
 rows=[]
 for n,r in sorted(data.items()):
  row={k:r[k] for k in ('solver','order','layers','dt_s','scenario','status','requested_hours','completed_hours','failure','wall_s','energy_error_J_kg','salt_relative_change','passive_relative_change','limiter')};row['case']=n;rows.append(row)
 comparisons=[]
 for left,right in [('budget','legacy'),('budget','refresh'),('budget','order5'),('dt600','dt75'),('budget','dt75'),('dt150','dt75')]:
  x,y=data[left],data[right]
  if x['status']==y['status']=='completed' and x['completed_hours']==y['completed_hours'] and x['layers']==y['layers']:
   comparisons.append(dict(left=left,right=right,metrics=compare(x,y)))
 a.output.write_text(json.dumps(dict(cases=rows,comparisons=comparisons,note='Numerical endpoint comparisons, not analytic references. Overlapping jobs: campaign wall times are not isolated performance measurements.'),indent=2)+'\n')
if __name__=='__main__':main()
