"""Analytic vector diagnostics on both global latitude layouts; CPU float64.

Unit sphere and unit amplitude make errors dimensionless. Each field is the
sum of a spherical-harmonic gradient and its clockwise tangent rotation.
For degree l, exact divergence = -l(l+1)q and curl = +l(l+1)q.
Stress uses the same spatial field with amplitude 0.1; it tests the linear
stress diagnostic, not a physical ocean response. No downloaded data required.
"""
import argparse
import json
from pathlib import Path
import torch
from vfs.geometry import EquiangularGrid


def measure(n, poles, mode):
    lat=torch.linspace(90.,-90.,n+1,dtype=torch.float64) if poles else torch.linspace(90.-90/n,-90.+90/n,n,dtype=torch.float64)
    lon=torch.arange(2*n,dtype=lat.dtype)*180/n
    g=EquiangularGrid.from_coordinates(lat,lon,radius=1.)
    p,l=map(torch.deg2rad,g.meshgrid())
    if mode=='axial':
        q=p.sin();gu=torch.zeros_like(q);gv=p.cos();eigenvalue=2
    elif mode=='tilted':
        q=p.cos()*l.cos();gu=-l.sin();gv=-p.sin()*l.cos();eigenvalue=2
    elif mode=='degree2':
        q=p.sin()*p.cos()*l.cos();gu=-p.sin()*l.sin();gv=(2*p).cos()*l.cos();eigenvalue=6
    else:
        raise ValueError(mode)
    u,v=gu+gv,gv-gu
    fields={'div':(g.div_from_collocated(u,v),-eigenvalue*q),
            'curl':(g.curl_from_collocated(u,v),eigenvalue*q),
            'stress_curl':(g.curl_from_collocated(.1*u,.1*v),.1*eigenvalue*q),
            'rotational_div':(g.div_from_collocated(gv,-gu),torch.zeros_like(q)),
            'irrotational_curl':(g.curl_from_collocated(gu,gv),torch.zeros_like(q))}
    row={'nlat':g.nlat,'nlon':g.nlon,'layout':'pole' if poles else 'center','mode':mode}
    for name,(actual,exact) in fields.items():
        err=actual-exact
        row[name]={'rms':float(g.mean(err.square()).sqrt()),
                   'polar_max':float(err[[0,-1]].abs().max()),
                   'polar_band_max':float(err[p.abs()>=torch.deg2rad(p.new_tensor(75.))].abs().max()),
                   'interior_max':float(err[1:-1].abs().max()),
                   'global_integral':float(g.integral(actual))}
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result={'scope':'CPU float64, analytic point-value references; polar values of FV diagnostics are cap averages',
            'torch':torch.__version__,
            'results':[measure(n,poles,mode) for poles in (False,True) for mode in ('axial','tilted','degree2') for n in (16,32,64,128)]}
    content=json.dumps(result,indent=2)
    if args.output:args.output.write_text(content+'\n')
    else:print(content)


if __name__=='__main__':main()
