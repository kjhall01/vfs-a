"""Optional real-data qualification; no WeatherBench2 dependency in vfs.

python run.py --weatherbench2 --output /path/to/results
python run.py --input snapshot.nc --output /path/to/results
Cloud access occurs only with an explicit input URI or --weatherbench2.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import torch
import xarray as xr
from vfs.geometry import EquiangularGrid

WB2 = 'https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
WB_RADIUS = 6367500.
TIMES = ['2020-01-15T12:00:00','2020-04-15T12:00:00','2020-07-15T12:00:00','2020-10-15T12:00:00']


def tensor(x):
    if isinstance(x,torch.Tensor):
        return x.detach().to(dtype=torch.float64,device="cpu").clone()
    return torch.from_numpy(np.array(x,dtype=np.float64,copy=True))


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def stats(actual, reference, area, mask=None):
    """Area-weighted comparison; undefined references are counted, never filled."""
    valid = np.isfinite(actual) & np.isfinite(reference)
    if mask is not None: valid &= mask
    count=int(valid.sum())
    if not count: return {'count':0}
    a,b,w=actual[valid],reference[valid],area[valid]
    w=w/w.sum();error=a-b
    am,bm=(w*a).sum(),(w*b).sum()
    av,bv=(w*(a-am)**2).sum(),(w*(b-bm)**2).sum()
    return {'count':count,'bias':float((w*error).sum()),
            'rms':float(np.sqrt((w*error**2).sum())),
            'reference_rms':float(np.sqrt((w*b**2).sum())),
            'max_abs':float(np.abs(error).max()),
            'correlation':float((w*(a-am)*(b-bm)).sum()/np.sqrt(av*bv)) if av*bv>0 else None}


def compare(g, actual, reference):
    lat,lon=map(np.asarray,g.meshgrid());area=np.asarray(g.cell_area)
    masks={'global':np.ones(g.shape,dtype=bool),'below_85deg':np.abs(lat)<85,
           'polar_band':np.abs(lat)>=75,'poles':np.abs(lat)==90,
           'seam_columns':np.broadcast_to((np.arange(g.nlon)<2)|(np.arange(g.nlon)>=g.nlon-2),g.shape)}
    return {name:stats(actual,reference,area,mask) for name,mask in masks.items()}


def references(g,u,v):
    """Independent NumPy spherical FD plus replay of current WB2 formulas.

    This reference has no finite-volume conservation claim. Latitude derivatives
    are second-order finite differences; longitude is periodic in the spherical
    reference. Pole point derivatives are undefined and explicitly NaN.
    WB2 replay uses its nonperiodic longitude derivative and omits metric terms.
    """
    lat=np.deg2rad(np.asarray(g.lat_deg));dl=2*np.pi/g.nlon
    cos=np.cos(lat)[:,None];safe=np.where(np.abs(cos)>1e-10,cos,np.nan)
    dx=lambda x:(np.roll(x,-1,axis=1)-np.roll(x,1,axis=1))/(2*dl)
    dy=lambda x:np.gradient(x,lat,axis=0,edge_order=2)
    sphere_div=(dx(u)+dy(v*cos))/(g.radius*safe)
    sphere_curl=(dx(v)-dy(u*cos))/(g.radius*safe)
    # Match WB2's documented implementation rather than calling its package.
    wb_dx=lambda x:np.where(np.abs(cos)>1e-6,np.gradient(x,dl,axis=1,edge_order=1)/np.where(np.abs(cos)>1e-6,cos,1)/WB_RADIUS,0.)
    wb_dy=lambda x:np.gradient(x,lat,axis=0,edge_order=1)/WB_RADIUS
    return {'spherical_fd':(sphere_div,sphere_curl),'wb2_formula_replay':(wb_dx(u)+wb_dy(v),wb_dx(v)-wb_dy(u))}


def closure(integral,boundary,scale):
    residual=float(integral-boundary)
    relative=abs(residual)/max(float(scale),1e-30)
    if relative>2e-12: raise AssertionError(f'discrete budget failed: {relative}')
    return {'integral':float(integral),'boundary':float(boundary),'residual':residual,'relative':relative}


def region_masks(lat,lon):
    lon=lon%360
    return {'global':torch.ones_like(lat,dtype=torch.bool),
            'dateline_box':(lat>=-30)&(lat<=30)&(lon>=150)&(lon<=210),
            'seam_box':(lat>=20)&(lat<=60)&((lon>=330)|(lon<=30)),
            'north_cap':lat>=75,'south_cap':lat<=-75}


def budgets(g,u,v,phi=None):
    """Compare volume sums to boundary-only incidence sums, including caps."""
    e,n=g.wind_to_faces(tensor(u),tensor(v));div=g.div(e,n);curl=g.curl(e,n)
    E,N=e*g.east_face_length,n*g.north_face_length
    result={'divergence':{},'curl':{}}
    for name,m in region_masks(*g.meshgrid()).items():
        m=m.double();zero=torch.zeros_like(m[:1])
        boundary_e=(m-m.roll(-1,-1))*E
        boundary_n=g.latitude_direction*(torch.cat((zero,m))-torch.cat((m,zero)))*N
        terms=m*div*g.cell_area
        result['divergence'][name]=closure(terms.sum(),boundary_e.sum()+boundary_n.sum(),terms.abs().sum()+boundary_e.abs().sum()+boundary_n.abs().sum())
    lat=g.lat_bounds_deg[:,None].expand(g.nlat+1,g.nlon)
    lon=g.lon_bounds_deg[1:][None,:].expand_as(lat)
    e_line=e*g.east_distance;n_line=n[1:-1]*g.north_distance[1:-1]
    for name,m in region_masks(lat,lon).items():
        m=m.double();middle=m[1:-1];zero=torch.zeros_like(m[:1])
        edge_coefficient=g.latitude_direction*(torch.cat((middle,zero))-torch.cat((zero,middle)))
        edge_coefficient[0]-=g.latitude_direction*m[0].mean()
        edge_coefficient[-1]+=g.latitude_direction*m[-1].mean()
        boundary_e=edge_coefficient*e_line
        boundary_n=(middle.roll(1,-1)-middle)*n_line
        terms=m*curl*g.corner_area
        result['curl'][name]=closure(terms.sum(),boundary_e.sum()+boundary_n.sum(),terms.abs().sum()+boundary_e.abs().sum()+boundary_n.abs().sum())
    if phi is not None:
        q=tensor(phi);ge,gn=g.grad(q)
        circulation=g.curl(ge,gn)*g.corner_area
        result['geopotential_curl_grad']={'max_circulation_residual':float(circulation.abs().max()),
          'relative_to_scalar_range':float(circulation.abs().max()/max(float(q.max()-q.min()),1e-30))}
        if result['geopotential_curl_grad']['relative_to_scalar_range']>2e-12:raise AssertionError('curl grad failed')
        work=g.integral(q*g.laplacian(q));norm=g.face_inner(ge,gn,ge,gn)
        result['geopotential_adjoint']=closure(work,-norm,abs(work)+abs(norm))
    return result


def invariance(g,u,v):
    baseline=(g.div_from_collocated(tensor(u),tensor(v)),g.curl_from_collocated(tensor(u),tensor(v)))
    results={}
    for mode in ('reverse_latitude','roll_longitude','longitude_convention'):
        lat,lon=g.lat_deg.numpy().copy(),g.lon_deg.numpy().copy();a,b=u,v
        if mode=='reverse_latitude':lat=lat[::-1];a,b=u[::-1],v[::-1]
        elif mode=='roll_longitude':lon=np.roll(lon,7);a,b=np.roll(u,7,-1),np.roll(v,7,-1)
        else:lon=(lon+180)%360-180
        other=EquiangularGrid.from_coordinates(lat,lon,radius=g.radius)
        out=(other.div_from_collocated(tensor(a),tensor(b)),other.curl_from_collocated(tensor(a),tensor(b)))
        errors=[]
        for x,y in zip(out,baseline):
            if mode=='reverse_latitude':x=x.flip(0)
            elif mode=='roll_longitude':x=x.roll(-7,-1)
            relative=float((x-y).abs().max())/max(float(y.abs().max()),1e-30)
            if relative>2e-12:raise AssertionError(f'coordinate invariance failed: {mode}')
            errors.append(relative)
        results[mode]={'divergence_relative_max':errors[0],'curl_relative_max':errors[1]}
    return results


def overlap_weights(source,target):
    """Separable exact spherical area overlaps; arrays retain axis order."""
    sb,tb=source.lat_bounds_deg.deg2rad().sin(),target.lat_bounds_deg.deg2rad().sin()
    sl,sh=torch.minimum(sb[:-1],sb[1:]),torch.maximum(sb[:-1],sb[1:])
    tl,th=torch.minimum(tb[:-1],tb[1:]),torch.maximum(tb[:-1],tb[1:])
    lat=(torch.minimum(th[:,None],sh)-torch.maximum(tl[:,None],sl)).clamp_min(0)/(th-tl)[:,None]
    sb,tb=source.lon_bounds_deg,target.lon_bounds_deg
    # Align the unwrapped source ring to the target's starting turn.
    sb=sb-360*torch.floor((sb[0]-tb[0])/360)
    lon=torch.zeros(target.nlon,source.nlon,dtype=torch.float64)
    for shift in (-360.,0.,360.):
        lon+=(torch.minimum(tb[1:,None],sb[None,1:]+shift)-torch.maximum(tb[:-1,None],sb[None,:-1]+shift)).clamp_min(0)
    lon/=torch.diff(tb)[:,None]
    return lat,lon


def average(weights,value):
    lat,lon=weights;x=tensor(value);valid=torch.isfinite(x)
    result=lat@torch.where(valid,x,0.)@lon.T
    coverage=lat@valid.double()@lon.T
    return torch.where(coverage>1-1e-10,result,torch.nan)


def vector_average(source,target,weights,u,v):
    """Average Cartesian vectors, then project on target tangent directions.

    This avoids averaging incompatible polar/local bases. Tangent projection
    is not an energy- or momentum-conserving remap qualification.
    """
    p,l=map(torch.deg2rad,source.meshgrid());u,v=tensor(u),tensor(v)
    xyz=(-u*l.sin()-v*p.sin()*l.cos(),u*l.cos()-v*p.sin()*l.sin(),v*p.cos())
    x,y,z=[average(weights,a) for a in xyz]
    p,l=map(torch.deg2rad,target.meshgrid())
    return -x*l.sin()+y*l.cos(),-x*p.sin()*l.cos()-y*p.sin()*l.sin()+z*p.cos()


def coarse_checks(g,u,v,diagnostics,refs):
    result={}
    for nlat,nlon,poles in ((32,64,False),(121,240,True)):
        if nlat>=g.nlat or nlon>=g.nlon:continue
        target=EquiangularGrid.from_coordinates(torch.linspace(-90,90,nlat,dtype=torch.float64),torch.arange(nlon,dtype=torch.float64)*360/nlon,radius=g.radius) if poles else EquiangularGrid(nlat,nlon,radius=g.radius)
        weights=overlap_weights(g,target);cu,cv=vector_average(g,target,weights,u,v)
        computed=(target.div_from_collocated(cu,cv),target.curl_from_collocated(cu,cv))
        name=f'{nlat}x{nlon}';result[name]={}
        for k,key in enumerate(('divergence','curl')):
            fine_mean=average(weights,diagnostics[k])
            result[name][key]={'differentiate_vs_average':compare(target,np.asarray(computed[k]),np.asarray(fine_mean)),
                'common_scale_references':{r:compare(target,np.asarray(fine_mean),np.asarray(average(weights,fields[k]))) for r,fields in refs.items()}}
    return result


def plot_case(out,g,diagnostics,refs,label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lat=np.asarray(g.lat_deg);lon=(np.asarray(g.lon_deg)+180)%360-180;order=np.argsort(lon)
    # Render a subsample only; all metrics use full-resolution fields.
    stride=max(1,g.nlon//480);order=order[::stride];rows=np.arange(0,g.nlat,max(1,g.nlat//240))
    ref=refs['spherical_fd'];fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    for i,name in enumerate(('Divergence','Vorticity')):
        fields=(diagnostics[i],ref[i],diagnostics[i]-ref[i])
        scale=np.nanquantile(np.abs(np.concatenate([fields[0].ravel(),fields[1].ravel()])),.99)*1e5
        for j,field in enumerate(fields):
            a=axes[i,j];lim=scale if j<2 else np.nanquantile(np.abs(field),.99)*1e5
            lim=max(lim,1e-12)
            im=a.pcolormesh(lon[order],lat[rows],field[np.ix_(rows,order)]*1e5,shading='nearest',cmap='RdBu_r',vmin=-lim,vmax=lim)
            a.set(title=f'{name}: '+['VFS','spherical FD','difference'][j],xlabel='Longitude (°)',ylabel='Latitude (°)')
            fig.colorbar(im,ax=a,label='10⁻⁵ s⁻¹')
    fig.suptitle(label+'; colors clipped at 99th percentile; full-grid metrics in JSON')
    fig.savefig(out,dpi=110,format='webp',pil_kwargs={'quality':65});plt.close(fig)
    # Row statistics expose polar behavior hidden by global area weighting.
    profiles=[]
    for i in range(2):
        difference=diagnostics[i]-ref[i]
        finite=np.isfinite(difference)
        count=finite.sum(axis=1)
        rms=np.sqrt(np.divide(np.where(finite,difference**2,0.).sum(axis=1),count,
                             out=np.full(len(lat),np.nan),where=count>0))
        profiles.append(rms)
    np.savetxt(out.with_suffix('.csv'),np.column_stack((lat,*profiles)),delimiter=',',
               header='latitude_deg,divergence_rms_s-1,vorticity_rms_s-1',comments='')
    fig,axes=plt.subplots(1,2,figsize=(10,3),constrained_layout=True)
    for i,name in enumerate(('Divergence','Vorticity')):
        axes[i].semilogy(lat,profiles[i]);axes[i].set(xlabel='Latitude (°)',ylabel='RMS difference (s⁻¹)',title=name+' vs spherical FD',xlim=(-90,90))
        axes[i].grid(alpha=.25)
    fig.suptitle(label+'; pole point references are undefined')
    fig.savefig(out.with_name(out.stem+'-latitude.webp'),dpi=100,format='webp',pil_kwargs={'quality':65});plt.close(fig)


def evaluate(ds,args,output):
    latitude=ds[args.lat].values;longitude=ds[args.lon].values
    g=EquiangularGrid.from_coordinates(latitude,longitude,radius=args.radius)
    if g.nlat<3:raise ValueError('independent FD comparison requires at least three latitudes')
    report={'source':args.input or WB2,'radius_m':g.radius,'shape':g.shape,
      'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'xarray':xr.__version__},
      'coordinate_hashes':{'latitude':digest(latitude),'longitude':digest(longitude)},
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'operator_sha256':hashlib.sha256(Path(__import__('inspect').getfile(EquiangularGrid)).read_bytes()).hexdigest(),
      'reference_warning':'WB2 fields are comparison products, not exact spherical truth. Current WB2 source omits curvature terms and uses nonperiodic longitude differences; archive derivation version is not assumed.',
      'cases':[]}
    time_values=args.times if args.times else (TIMES if args.weatherbench2 else list(ds.time.values[:1]) if 'time' in ds.dims else [None])
    levels=args.levels if 'level' in ds.dims else [None]
    names=[args.u,args.v]+[name for name in (args.geopotential,args.divergence,args.vorticity) if name in ds]
    for time in time_values:
        # Load one timestamp at a time; Zarr chunks may contain ALL 13 levels
        # even when only two levels are selected. No full-history .load().
        snapshot=ds[names].sel(time=time) if time is not None else ds[names]
        if levels!=[None]:snapshot=snapshot.sel(level=levels)
        snapshot=snapshot.load()
        for level in levels:
            case=snapshot.sel(level=level) if level is not None else snapshot
            read=lambda name:np.asarray(case[name].transpose(args.lat,args.lon).values,dtype=np.float64)
            u,v=read(args.u),read(args.v)
            phi=read(args.geopotential) if args.geopotential in case else None
            if not all(np.isfinite(a).all() for a in (u,v) if a is not None):raise ValueError('winds must be finite globally; no land/NaN filling is performed')
            if phi is not None and not np.isfinite(phi).all():raise ValueError('geopotential must be finite for identity tests')
            native=(np.asarray(g.div_from_collocated(tensor(u),tensor(v))),np.asarray(g.curl_from_collocated(tensor(u),tensor(v))))
            refs=references(g,u,v)
            if args.divergence in case and args.vorticity in case:refs['archive']=(read(args.divergence),read(args.vorticity))
            item={'time':str(time),'level_hpa':None if level is None else int(level),
                  'input_hashes':{name:digest(read(name)) for name in names},
                  'comparisons':{name:{key:compare(g,native[i],fields[i]) for i,key in enumerate(('divergence','curl'))} for name,fields in refs.items()},
                  'budgets':budgets(g,u,v,phi),'coordinate_invariance':invariance(g,u,v)}
            if 'archive' in refs:
                item['archive_vs_wb2_replay']={key:compare(g,refs['archive'][i],refs['wb2_formula_replay'][i]) for i,key in enumerate(('divergence','curl'))}
            if not args.no_coarse:item['coarse_scale']=coarse_checks(g,u,v,native,refs)
            stem=f'case-{len(report['cases'])+1:02d}'
            (output/f'{stem}.json').write_text(json.dumps(item,indent=2,allow_nan=False)+'\n')
            label=f'{str(time)} / {level} hPa'
            if not args.no_plots:plot_case(output/f'{stem}.webp',g,native,refs,label)
            report['cases'].append({'time':str(time),'level_hpa':item['level_hpa'],
                                   'metrics_file':f'{stem}.json',
                                   'plot_file':None if args.no_plots else f'{stem}.webp',
                                   'latitude_plot_file':None if args.no_plots else f'{stem}-latitude.webp',
                                   'latitude_csv_file':None if args.no_plots else f'{stem}.csv'})
            # Small checkpoints keep per-file evidence reviewable and uploadable.
            (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(f'Completed {label}',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input',help='Local NetCDF/Zarr path or explicit cloud Zarr URI')
    source.add_argument('--weatherbench2',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--times',nargs='+')
    parser.add_argument('--levels',nargs='+',type=int,default=[250,500])
    parser.add_argument('--radius',type=float,default=WB_RADIUS)
    parser.add_argument('--lat',default='latitude');parser.add_argument('--lon',default='longitude')
    parser.add_argument('--u',default='u_component_of_wind');parser.add_argument('--v',default='v_component_of_wind')
    parser.add_argument('--geopotential',default='geopotential')
    parser.add_argument('--divergence',default='divergence');parser.add_argument('--vorticity',default='vorticity')
    parser.add_argument('--no-plots',action='store_true');parser.add_argument('--no-coarse',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    path=WB2 if args.weatherbench2 else args.input
    cloud=path.startswith(('gs://','https://','s3://'))
    if cloud or path.rstrip('/').endswith('.zarr'):
        options={'storage_options':{'token':'anon'}} if path.startswith('gs://') else ({'storage_options':{'client_kwargs':{'trust_env':True}}} if path.startswith('https://') else {})
        ds=xr.open_zarr(path,consolidated=True,chunks=None,**options)
    else:ds=xr.open_dataset(path)
    try:evaluate(ds,args,args.output)
    finally:ds.close()


if __name__=='__main__':main()
