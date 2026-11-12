"""Optional local ERA5 pole sensitivity audit; never edits the input dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import xarray as xr
from vfs.geometry import EquiangularGrid
# Direct execution under python -I must not depend on import path injection.
import importlib.util
spec=importlib.util.spec_from_file_location('polar',Path(__file__).with_name('polar.py'))
polar=importlib.util.module_from_spec(spec);spec.loader.exec_module(polar)


def audit(path):
    ds=xr.open_dataset(path)
    u=ds.u_component_of_wind.values.squeeze().astype(float)
    v=ds.v_component_of_wind.values.squeeze().astype(float)
    lat=ds.latitude.values.astype(float);lon=ds.longitude.values.astype(float)
    g=EquiangularGrid.from_coordinates(lat,lon,radius=6367500.)
    if not g.pole_centered or u.shape != g.shape or v.shape != g.shape:
        raise ValueError("Expected one global pole-inclusive 2D wind snapshot")
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("Wind values must be finite")
    p=np.deg2rad(lat)[:,None];l=np.deg2rad(lon)[None,:]
    east=np.stack(np.broadcast_arrays(-np.sin(l),np.cos(l),np.zeros_like(p)),-1)
    north=np.stack(np.broadcast_arrays(-np.sin(p)*np.cos(l),-np.sin(p)*np.sin(l),np.cos(p)),-1)
    xyz=u[...,None]*east+v[...,None]*north
    rows=[];uc=u.copy();vc=v.copy()
    for i,j in ((0,1),(-1,-2)):
        mean=xyz[j].mean(0);mean[2]=0.
        rows.append({'latitude':float(lat[i]),'pole_cartesian_scatter_m_s':float(np.sqrt(np.mean(np.sum((xyz[i]-xyz[i].mean(0))**2,axis=-1)))),
                     'adjacent_ring_mean_cartesian_speed_m_s':float(np.linalg.norm(mean)),
                     'pole_u_range_m_s':[float(u[i].min()),float(u[i].max())],
                     'pole_v_range_m_s':[float(v[i].min()),float(v[i].max())]})
        # Deliberate sensitivity intervention, not an asserted true pole wind.
        uc[i]=(east[i]*mean).sum(-1);vc[i]=(north[i]*mean).sum(-1)
    metrics={}
    for label,uu,vv in [('original',u,v),('adjacent_ring_cartesian_mean_at_poles',uc,vc)]:
        a,b=g.wind_to_faces(polar.tensor(uu),polar.tensor(vv))
        div=g.div(a,b).numpy();curl=g.corners_to_cells(g.curl(a,b)).numpy()
        # numpy_fd assumes unit radius, scale to this dataset radius.
        fd,fc=polar.numpy_fd(g,uu,vv);fd/=g.radius;fc/=g.radius
        metrics[label]={}
        for name,x,y in [('divergence',div,fd),('curl',curl,fc)]:
            metrics[label][name]=polar.error(x,y,g.cell_area.numpy(),lat,[1,-2])
    return {'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'time':str(ds.time.values),'level_hPa':int(ds.level.values),
            'radius_m':g.radius,'pole_vectors':rows,'vfs_minus_spherical_fd_s_inverse':metrics,
            'warning':'Sensitivity only: neither corrected pole values nor finite differences are analytical truth.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(audit(a.input),indent=2)+'\n')


if __name__=='__main__':main()
