"""Analytic polar qualification: point, cell-average and dual-average references.

All analytic modes use unit radius and unit potential amplitude. Exact face
averages are independent audit inputs, not replacements for public adapters.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from vfs.geometry import EquiangularGrid


def avg_sin(a,b):
    return np.sin((a+b)/2)*np.sinc((b-a)/(2*np.pi))


def avg_cos(a,b):
    return np.cos((a+b)/2)*np.sinc((b-a)/(2*np.pi))


def components(mode, p, l, pw=0., lw=0.):
    """Gradient of q, averaged in phi/lambda over symmetric intervals."""
    sinp=np.sin(p)*np.sinc(pw/(2*np.pi));cosp=np.cos(p)*np.sinc(pw/(2*np.pi))
    sinl=np.sin(l)*np.sinc(lw/(2*np.pi));cosl=np.cos(l)*np.sinc(lw/(2*np.pi))
    if mode=='x':return np.broadcast_arrays(-sinl,-sinp*cosl)
    if mode=='z':return np.broadcast_arrays(np.zeros_like(l),cosp)
    if mode=='xz':return np.broadcast_arrays(-sinp*sinl,np.cos(2*p)*np.sinc(pw/np.pi)*cosl)
    raise ValueError(mode)


def q_point(mode,p,l):
    if mode=='x':return np.cos(p)*np.cos(l)
    if mode=='z':return np.sin(p)+np.zeros_like(l)
    if mode=='xz':return np.sin(p)*np.cos(p)*np.cos(l)
    raise ValueError(mode)


def q_average(mode, lat_edges, lon_edges, cap=False):
    """Closed-form area mean of q on latitude/longitude rectangles.

    Stabilized sin(w)-w avoids cancellation in small near-pole cells.
    Collapsed zero-area dual endpoints return zero placeholders.
    """
    lo=np.minimum(lat_edges[:-1],lat_edges[1:]);hi=np.maximum(lat_edges[:-1],lat_edges[1:])
    w=hi-lo;m=(lo+hi)/2;area=2*np.cos(m)*np.sin(w/2)
    if mode=='z':factor=(np.sin(lo)+np.sin(hi))/2
    elif mode=='x':
        delta=np.where(abs(w)<.1,-w**3/6+w**5/120-w**7/5040+w**9/362880,np.sin(w)-w)
        integral=w*np.cos(m)**2+np.cos(2*m)*delta/2
        factor=np.divide(integral,area,out=np.zeros_like(area),where=area>0)
    elif mode=='xz':factor=np.tan(m)*(np.cos(lo)**2+np.cos(lo)*np.cos(hi)+np.cos(hi)**2)/3
    else:raise ValueError(mode)
    factor=np.where(w>0,factor,0.)
    lon=np.ones(len(lon_edges)-1) if mode=='z' else avg_cos(lon_edges[:-1],lon_edges[1:])
    result=factor[:,None]*lon[None,:]
    if cap:result[[0,-1]]=result[[0,-1]].mean(axis=1,keepdims=True)
    return result


def analytic(g,mode,rotation=False):
    lat=np.deg2rad(g.lat_deg.numpy());lon=np.deg2rad(g.lon_deg.numpy());dl=2*np.pi/g.nlon
    lb=np.deg2rad(g.lat_bounds_deg.numpy());ob=np.deg2rad(g.lon_bounds_deg.numpy())
    transform=lambda a,b:(b,-a) if rotation else (a,b)
    u,v=transform(*components(mode,lat[:,None],lon[None,:]))
    # Exact normal-flux line means on primal cell boundaries.
    flux_e=transform(*components(mode,((lb[:-1]+lb[1:])/2)[:,None],ob[1:][None,:],np.abs(np.diff(lb))[:,None]))[0]
    flux_n=transform(*components(mode,lb[:,None],((ob[:-1]+ob[1:])/2)[None,:],lw=np.diff(ob)[None,:]))[1]
    # Exact tangential-circulation line means on dual cell boundaries.
    circ_e=transform(*components(mode,lat[:,None],(lon+dl/2)[None,:],lw=dl))[0]
    circ_n=np.zeros((g.nlat+1,g.nlon))
    circ_n[1:-1]=transform(*components(mode,((lat[:-1]+lat[1:])/2)[:,None],lon[None,:],np.abs(np.diff(lat))[:,None]))[1]
    eigenvalue=6 if mode=='xz' else 2
    point=q_point(mode,lat[:,None],lon[None,:])
    cell=q_average(mode,lb,ob,cap=g.pole_centered)
    dual_lat=np.r_[lb[0],lat,lb[-1]];dual_lon=np.r_[lon,lon[-1]+dl]
    dual=q_average(mode,dual_lat,dual_lon,cap=not g.pole_centered)
    div=-eigenvalue*cell if not rotation else np.zeros_like(cell)
    curl=eigenvalue*dual if rotation else np.zeros_like(dual)
    curl_cell=eigenvalue*cell if rotation else np.zeros_like(cell)
    return dict(u=u,v=v,flux=(flux_e,flux_n),circulation=(circ_e,circ_n),div=div,curl=curl,curl_cell=curl_cell,
                div_point=-eigenvalue*point if not rotation else np.zeros_like(point),
                curl_point=eigenvalue*point if rotation else np.zeros_like(point))


def numpy_fd(g,u,v):
    lat=np.deg2rad(g.lat_deg.numpy());cos=np.cos(lat)[:,None];dl=2*np.pi/g.nlon
    dx=lambda f:(np.roll(f,-1,1)-np.roll(f,1,1))/(2*dl)
    dy=lambda f:np.gradient(f,lat,axis=0,edge_order=2)
    denominator=np.where(abs(cos)>1e-10,cos,np.nan)
    return (dx(u)+dy(v*cos))/denominator,(dx(v)-dy(u*cos))/denominator


def tensor(x):return torch.from_numpy(np.array(x,dtype=np.float64,copy=True))


def error(actual,exact,area,lat,polar_rows):
    actual=np.asarray(actual);d=actual-exact;valid=np.isfinite(d)&(area>0)
    last=np.zeros_like(valid);last[polar_rows]=True
    masks={'global':valid,'last_nonpolar_ring':valid&last,
           'above_75':valid&(np.abs(lat[:,None])>=75),
           'pole_caps':valid&(np.abs(lat[:,None])==90)}
    out={}
    for name,mask in masks.items():
        if mask.any():out[name]={'rms':float(np.sqrt(np.sum(area[mask]*d[mask]**2)/np.sum(area[mask]))),'max':float(np.abs(d[mask]).max())}
    return out


def measure(n,poles,mode,rotation):
    lat=np.linspace(90,-90,n+1) if poles else np.linspace(90-90/n,-90+90/n,n)
    lon=np.arange(2*n)*180/n
    g=EquiangularGrid.from_coordinates(lat,lon,radius=1.)
    a=analytic(g,mode,rotation);u,v=map(tensor,(a['u'],a['v']));e,north=g.wind_to_faces(u,v)
    div=g.div(e,north);curl=g.curl(e,north);cellcurl=g.corners_to_cells(curl)
    exact_div=g.div(*map(tensor,a['flux']));exact_curl=g.curl(*map(tensor,a['circulation']))
    fd_div,fd_curl=numpy_fd(g,a['u'],a['v'])
    cellerror=lambda x,y:error(x,y,g.cell_area.numpy(),lat,[1,-2] if poles else [0,-1])
    dualerror=lambda x,y:error(x,y,g.corner_area.numpy(),g.lat_bounds_deg.numpy(),[1,-2])
    return {'intervals':n,'layout':'poles' if poles else 'centers','mode':mode,'field':'rotation' if rotation else 'gradient',
      'vfs_div_vs_cell_mean':cellerror(div,a['div']),
      'vfs_native_curl_vs_dual_mean':dualerror(curl,a['curl']),
      'vfs_cell_curl_vs_cell_mean':cellerror(cellcurl,a['curl_cell']),
      'fd_div_vs_point':cellerror(fd_div,a['div_point']),
      'fd_curl_vs_point':cellerror(fd_curl,a['curl_point']),
      'fd_div_vs_cell_mean':cellerror(fd_div,a['div']),
      'fd_curl_vs_cell_mean':cellerror(fd_curl,a['curl_cell']),
      'exact_flux_div_vs_cell_mean':cellerror(exact_div,a['div']),
      'exact_circulation_curl_vs_dual_mean':dualerror(exact_curl,a['curl']),
      'dual_to_cell_representation_only':cellerror(g.corners_to_cells(tensor(a['curl'])),a['curl_cell'])}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--intervals',nargs='+',type=int,default=[16,32,64,128,256,720]);args=p.parse_args()
    # One file per resolution keeps each artifact below the upload limit.
    args.output.mkdir(parents=True,exist_ok=True)
    for n in args.intervals:
        rows=[measure(n,poles,mode,rotation) for poles in (False,True) for mode in ('x','z','xz') for rotation in (False,True)]
        (args.output/f'analytic-{n}.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        print(n,flush=True)


if __name__=='__main__':main()
