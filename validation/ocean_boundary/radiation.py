"""Analytic two-band layer absorption and independent Hamiltonian increments."""
import json
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean,OceanSurfaceHeating


def run():
    mesh=VoronoiMesh(n_cells=32,lloyd_iters=2,rotation_scheme='compatible')
    rows=[]
    for n in (1,4,16,64):
        ocean=StratifiedOcean(mesh,OceanColumn(n,total_depth=40.))
        h=torch.full((mesh.nCells,n),40./n,dtype=torch.float64)
        s=ocean.state(u_e=torch.zeros(mesh.nEdges,n,dtype=h.dtype),h=h,
            temperature=torch.full_like(h,12.),salinity=torch.full_like(h,35.))
        area=ocean.energy.mass_area(s)/ocean.column.rho0
        mass=(ocean.energy.mass_area(s)*h).sum()
        z=torch.linspace(0.,40.,n+1,dtype=h.dtype)
        f=200*(.6*torch.exp(-z/.5)+.4*torch.exp(-z/20.))
        for bottom in ('export','absorb'):
            heating=OceanSurfaceHeating(.6,(.5,20.),bottom=bottom)
            out,r=heating.apply(ocean,s,3600.,shortwave=torch.full_like(h[:,0],200.),surface_heat=torch.full_like(h[:,0],-50.))
            expected=f[:-1]-f[1:]
            expected[0]-=50.
            if bottom=='absorb':expected[-1]+=f[-1]
            expected=3600*area*expected[None,:]
            relative=float((r.layer_heat-expected).abs().max()/expected.abs().max())
            full_error=float((ocean.budgets(out)['total']-ocean.budgets(s)['total']-r.external_heat)/mass)
            rows.append(dict(layers=n,bottom=bottom,relative_layer_flux_error=relative,
                net_water_heat_W_m2=float(r.external_heat/(3600*area.sum())),
                bottom_export_W_m2=float(r.bottom_exported_energy.sum()/(3600*area.sum())),
                full_H_residual_J_kg=full_error,
                maximum_layer_H_residual_J_kg=float((r.layer_energy_residual/(ocean.energy.mass_area(s)*h)).abs().max())))
            assert relative<5e-15
            assert abs(full_error)<1e-8
    return dict(scope='CPU float64, 40 m column, 200 W/m2 shortwave and -50 W/m2 surface heat for 3600 s; exact analytic layer-average absorption, not a calibrated optical profile',results=rows)

if __name__=='__main__':print(json.dumps(run(),indent=2))
