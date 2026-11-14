"""Independent GSW density-derivative comparison on a declared CT/SA/p grid.

Requires optional GSW for this offline diagnostic only. Roquet uses geopotential
height z=-p metres (the paper's 1 m/dbar approximation), not in-situ density fit.
"""
import argparse,json,sys
from pathlib import Path
import torch
import gsw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ocean_column'))
from roquet import RoquetFreezingEOS
from linear import LinearEOS


def evaluate():
    c=RoquetFreezingEOS();l=LinearEOS();rows=[]
    for p in (0.,1000.,3000.,5000.):
        for t in (-1.5,0.,5.,15.,25.,30.):
            sa=35.;rho=gsw.rho(sa,t,p);a=float(rho*gsw.alpha(sa,t,p));b=float(rho*gsw.beta(sa,t,p))
            ra=-(c.t+2*c.tt*t+c.tz*p)
            rows.append(dict(CT_C=t,SA_g_kg=sa,p_dbar=p,gsw_minus_rho_CT=a,roquet_minus_rho_CT=ra,roquet_thermal_relative_error=ra/a-1,
                linear_thermal_relative_error=(1025*l.alpha)/a-1,roquet_haline_relative_error=c.s/b-1))
    return dict(gsw_version=gsw.__version__,reference='GSW rho*alpha and rho*beta at native CT/SA/p; geopotential z=-p m',rows=rows)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(evaluate(),indent=2)+'\n')
