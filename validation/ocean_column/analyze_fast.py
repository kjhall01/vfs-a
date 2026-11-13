"""Compact long-column evidence and comparisons with the retained baseline."""
import argparse,hashlib,json
from pathlib import Path
from analyze_sustained import compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=[];comparisons=[]
    for name,baseline in [('wind24','L64-wind24'),('cooling12','L64-cooling12'),('rain12','L64-rain_heating12')]:
        path=a.input/(name+'.json');r=json.loads(path.read_text());raw_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        row={k:v for k,v in r.items() if k!='profiles'}
        row.update(initial_profile=r['profiles'][0],final_profile=r['profiles'][-1],hourly=[{k:q[k] for k in ('time_s','energy_minus_input_J_kg','temperature_mld_m','limiter')} for q in r['profiles']],raw_sha256=raw_hash)
        text=json.dumps(row,indent=2)+'\n';assert len(text.encode())<=131072
        (a.output/(name+'.json')).write_text(text)
        manifest.append(dict(file=path.name,raw_bytes=path.stat().st_size,sha256=raw_hash,raw_omitted=path.stat().st_size>131072,representation='endpoint profiles plus hourly scalar diagnostics'))
        refpath=a.reference/(baseline+'.json');ref=json.loads(refpath.read_text())
        if r['status']=='completed' and ref['status']=='completed' and r['completed_hours']==ref['completed_hours']:
            comparisons.append(dict(case=name,reference=baseline,reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),metrics=compare(r,ref)))
    (a.output/'column-comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
