"""Same-environment output/initialization capture, not a scientific experiment."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import torch
import numpy as np
import xarray as xr
from vfs import State, VoronoiMesh, CharneyPhillipsColumn, make_hybrid_coefficients
from vfs.geometry.regrid import LatLonGrid, MeshToLatLonKNN
from vfs.data.topography import build_topographic_column, earth_surface_pressure_from_phi_s
from vfs.earth import earth_like_dry_initial_state
from vfs.field import Field
from vfs.boundary import synthetic_earth_boundary, write_boundary_netcdf, load_native_boundary_field


def capture():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
    a, b = make_hybrid_coefficients(3, p_top=5000.)
    column = CharneyPhillipsColumn(3, a, b, phi_s=mesh.cell_xyz[:, 0]*100.)
    wind = torch.linspace(-1., 1., mesh.nEdges*3).reshape(mesh.nEdges, 3).requires_grad_()
    ps = (1e5+100*mesh.cell_xyz[:, :1]).requires_grad_()
    theta = (300+torch.arange(4)[None, :]+mesh.cell_xyz[:, :1]).requires_grad_()
    state = State.from_theta(wind, theta, ps, mesh=mesh, column=column)
    grid = LatLonGrid.regular(nlat=8, nlon=16)
    regrid = MeshToLatLonKNN.from_mesh(mesh, grid)
    fields = regrid.state_to_latlon(state, fields='all')
    clean = lambda d: {k: (v.values.detach().clone() if isinstance(v, Field) else v.detach().clone() if isinstance(v, torch.Tensor) else v) for k, v in d.items()}
    result = {'atmosphere': clean(fields)}
    objective = fields['temperature_full'].values.square().sum()+fields['wind_speed'].values.sum()
    result['gradients'] = [g.detach().clone() for g in torch.autograd.grad(objective, (wind, theta, ps))]
    # Existing output contract only, not a numerical ocean implementation.
    h = torch.ones(mesh.nCells, 3)*10
    ocean = SimpleNamespace(h=h, temp_h=280*h, u_e=wind.detach(), mesh=mesh,
        eta=lambda: h.sum(-1, keepdim=True)-30,
        temperature=lambda: 280+torch.zeros_like(h),
        salinity=lambda: 35+torch.zeros_like(h),
        density=lambda: 1025+torch.zeros_like(h), tracers=lambda: None)
    result['ocean_output_contract'] = clean(regrid.state_to_latlon(ocean, fields='all'))
    with TemporaryDirectory() as directory:
        path=Path(directory)/'state.nc'
        state.to_netcdf(path)
        with xr.open_dataset(path) as ds:
            result['snapshot']={k:torch.as_tensor(ds[k].values.copy()) for k in ds.variables}
            result['snapshot_attrs']={k: (v.item() if isinstance(v, np.generic) else v) for k,v in ds.attrs.items()}
        restored=state.load_netcdf(path)
        result['restored']={k:getattr(restored,k) for k in ('u_e','mtheta_i','ps')}
    topographic=build_topographic_column(mesh,column.phi_s,n_levels=3)
    result['terrain']={'A':topographic.A_interface,'B':topographic.B_interface,
                       'ps':earth_surface_pressure_from_phi_s(column.phi_s)}
    initialized=earth_like_dry_initial_state(mesh,column,wind_noise=.1,seed=13)
    result['earth_initial']={k:getattr(initialized,k) for k in ('u_e','mtheta_i','ps')}
    boundary=synthetic_earth_boundary(grid)
    result['boundary']={k:v['values'] for k,v in boundary.items()}
    with TemporaryDirectory() as directory:
        path=Path(directory)/'boundary.nc'
        write_boundary_netcdf(path,grid,boundary,{'surface_geopotential':column.phi_s})
        loaded,_=load_native_boundary_field(path,'surface_geopotential',n_cells=mesh.nCells)
        result['loaded_boundary']=loaded
    return result


def compare(a,b,path='output'):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True,msg=lambda s:path+': '+s)
        return 1
    if isinstance(a,dict):
        # The ocean output fix adds one explicit convention label. Keep every
        # numerical comparison exact and reject any other metadata difference.
        if path == 'output/ocean_output_contract/attrs' and 'density_convention' not in b:
            assert a.get('density_convention') == 'legacy_state_linear_eos', path
            a = {k: v for k, v in a.items() if k != 'density_convention'}
        assert a.keys()==b.keys(),path
        return sum(compare(a[k],b[k],path+'/'+k) for k in a)
    if isinstance(a,list):
        assert len(a)==len(b)
        return sum(compare(x,y,path) for x,y in zip(a,b))
    assert a==b,(path,a,b)
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--save',type=Path)
    group.add_argument('--compare',type=Path)
    args=parser.parse_args()
    result=capture()
    if args.save:
        if args.save.exists():raise FileExistsError(args.save)
        torch.save(result,args.save)
    else:print('Exactly equal output tensors:',compare(result,torch.load(args.compare,weights_only=True)))
