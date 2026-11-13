"""Matched numerical-reference accuracy and explicit limiter/cost reporting."""
import argparse,json
from pathlib import Path
from analyze_sustained import compare

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 data={f.stem:json.loads(f.read_text()) for f in a.input.glob('*.json')}
 data={n:r for n,r in data.items() if 'profiles' in r}
 rows=[];comparisons=[]
 for n,r in sorted(data.items()):
  rows.append(dict(case=n,**{k:r.get(k) for k in ('method','funding','substeps','predictors','reduction','layers','dt_s','status','requested_hours','completed_hours','failure','wall_s','energy_error_J_kg','salt_relative_change','passive_relative_change','limiter')}))
 for name in ('dt300','dt150','dt75','dt18','sub4','sub16','reference300','factor05','factor05_fine','column_control','initial_local'):
  x,y=data[name],data['reference18']
  if x['status']==y['status']=='completed' and x['completed_hours']==y['completed_hours']:
   comparisons.append(dict(case=name,reference='reference18',metrics=compare(x,y)))
 comparisons.append(dict(case='factor05',reference='dt300',metrics=compare(data['factor05'],data['dt300'])))
 comparisons.append(dict(case='factor05_fine',reference='dt18',metrics=compare(data['factor05_fine'],data['dt18'])))
 a.output.write_text(json.dumps(dict(cases=rows,comparisons=comparisons,note='Coupled dt18.75 is a numerical reference, not an analytic solution or established converged truth. Subcycling changes only mixing, not outer forcing/rotation. Concurrent wall timings are not isolated benchmarks.'),indent=2)+'\n')
if __name__=='__main__':main()
