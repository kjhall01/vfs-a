"""Local buoyancy response comparison, independent of matched initialization.

Both density slopes are normalized by the same Boussinesq rho0=1025, not by
in-situ density. This measures response to T68, not a seawater standard alpha.
"""
import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ocean_column'))
from linear import LinearEOS
from vfs.ocean.jmd95_energy import jmd95_partials

def evaluate():
    c=LinearEOS();rows=[]
    for p in (0.,3000.):
        for t in (0.,5.,15.,25.,30.):
            s=torch.tensor(35.,dtype=torch.float64);theta=s*0+t;pressure=s*0+p
            rho,rt,rs,rp=jmd95_partials(s,theta,pressure)
            alpha=float(-rt/1025);beta=float(rs/1025)
            rows.append(dict(T68_C=t,pressure_dbar=p,jmd95_boussinesq_alpha=alpha,linear_alpha=c.alpha,
                thermal_response_relative_difference=c.alpha/alpha-1,jmd95_boussinesq_beta=beta,
                haline_response_relative_difference=c.beta/beta-1))
    return dict(reference='implemented JMD95 partials at S=35, fixed pressure; same rho0=1025',rows=rows)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(evaluate(),indent=2)+'\n')
