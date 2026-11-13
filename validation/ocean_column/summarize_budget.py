"""Retain signed final ledgers and scalar iteration trends, omit raw histories."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=a.input.read_bytes();r=json.loads(raw);summary={k:v for k,v in r.items() if k not in ('trajectory','replays')}
    summary['raw_provenance']=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),omitted='Full per-iteration layer arrays omitted; signed final ledgers and scalar trends retained.')
    if 'trajectory' in r:summary['trajectory']={k:v for k,v in r['trajectory'].items() if k!='profiles'}
    summary['replays']={}
    for name,c in r['replays'].items():
        out={k:v for k,v in c.items() if k!='history'}
        if 'history' in c:
            out['iteration_trend']=[dict(iteration=h['iteration'],error=h['error'],max_budget_residual=float(np.max(h['predicted']['scaled_residual'])),min_k=float(np.min(h['predicted']['k']))) for h in {v['iteration']:v for v in c['history']}.values()]
            ledger=c['predicted'];index=np.unravel_index(np.argmax(ledger['scaled_residual']),np.shape(ledger['k']))
            out['worst_layer']=dict(column=int(index[0]),layer_from_top_zero_based=int(index[1]),depth_m=np.array(c['depth'])[index],**{k:np.array(v)[index] for k,v in ledger.items()})
        summary['replays'][name]=out
    text=json.dumps(summary,indent=2)+'\n'
    if len(text.encode())>131072:raise ValueError('Summary exceeds artifact cap')
    a.output.write_text(text)
if __name__=='__main__':main()
