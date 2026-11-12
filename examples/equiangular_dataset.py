"""Diagnose a global regular dataset without changing coordinate order.

No arguments: synthetic 1-degree ERA5-style layout (181 x 360).
With a file: --file data.nc --u u10 --v v10 [--lat latitude --lon longitude]
Choose one time/level slice upstream for large files; this example loads all
selected variables into memory. For stress, pass eastward/northward stress
variable names; the resulting curl has stress-units/m, not s^-1.
"""
import argparse
import torch
import xarray as xr
from vfs.geometry import EquiangularGrid


def diagnose(u, v, *, lat='latitude', lon='longitude', radius=6.371e6):
    """Explicit xarray boundary; tensor operators preserve leading dimensions."""
    u,v=xr.align(u,v,join='exact')
    u=u.transpose(...,lat,lon)
    if set(u.dims)!=set(v.dims):
        raise ValueError('vector component dimensions must agree')
    v=v.transpose(*u.dims)
    grid=EquiangularGrid.from_coordinates(u[lat].values,u[lon].values,radius=radius)
    east=torch.as_tensor(u.values.copy(),dtype=torch.float64)
    north=torch.as_tensor(v.values.copy(),dtype=torch.float64)
    curl=grid.curl_from_collocated(east,north)
    div=grid.div_from_collocated(east,north)
    result=xr.Dataset({
        'curl':xr.DataArray(curl.detach().numpy(),dims=u.dims,coords=u.coords),
        'divergence':xr.DataArray(div.detach().numpy(),dims=u.dims,coords=u.coords),
    })
    result.attrs['radius_m']=radius
    result.attrs['polar_convention']='One cap per pole; repeated scalar outputs, fractional area weights'
    return result,grid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file')
    parser.add_argument('--u',default='u10')
    parser.add_argument('--v',default='v10')
    parser.add_argument('--lat',default='latitude')
    parser.add_argument('--lon',default='longitude')
    args=parser.parse_args()
    if args.file:
        with xr.open_dataset(args.file) as ds:
            u,v=ds[args.u].load(),ds[args.v].load()
    else:
        lat=torch.linspace(90.,-90.,181,dtype=torch.float64)
        lon=torch.arange(360,dtype=lat.dtype)
        p,l=torch.meshgrid(lat.deg2rad(),lon.deg2rad(),indexing='ij')
        u=xr.DataArray((-p.sin()*l.cos()).numpy(),dims=(args.lat,args.lon),coords={args.lat:lat.numpy(),args.lon:lon.numpy()})
        v=xr.DataArray(l.sin().numpy(),dims=u.dims,coords=u.coords)
    result,grid=diagnose(u,v,lat=args.lat,lon=args.lon)
    print(result)
    print('Area (m²):',float(grid.cell_area.sum()))
    print('Global curl integral:',grid.integral(torch.as_tensor(result['curl'].values)).numpy())


if __name__=='__main__':main()
