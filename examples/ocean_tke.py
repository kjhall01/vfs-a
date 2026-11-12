"""Closed stratified column shear -> TKE -> heat, using installed public APIs.

Illustrative coefficients, not calibrated ocean turbulence. CPU float64.
Run with python -I examples/ocean_tke.py after installing VFS.
"""
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import OceanColumn
from vfs.ocean.stratified import StratifiedOcean, TKEOcean, TKEClosure


def main():
    torch.set_num_threads(1)
    mesh=VoronoiMesh(n_cells=32,lloyd_iters=2,rotation_scheme='compatible')
    ocean=StratifiedOcean(mesh,OceanColumn(16,total_depth=100.))
    model=TKEOcean(ocean,closure=TKEClosure(mixing_length=2.))
    h=torch.full((mesh.nCells,16),100./16,dtype=torch.float64)
    resolved=ocean.state(h=h,
        u_e=torch.linspace(.3,-.3,16,dtype=h.dtype).expand(mesh.nEdges,16),
        temperature=torch.linspace(15.,12.,16,dtype=h.dtype).expand_as(h),
        salinity=torch.full_like(h,35.))
    state=model.state(resolved,tke=torch.full_like(h,.01))  # explicit seed, m²/s²
    before=model.budgets(state)
    with torch.no_grad():
        for _ in range(20):
            state,report=model.mix_with_report(state,10.)
    after=model.budgets(state)
    mass=(ocean.energy.mass_area(resolved)*h).sum()
    print('Resolved KE change (J/kg):',float((after['kinetic']-before['kinetic'])/mass))
    print('TKE change (J/kg):',float((after['tke']-before['tke'])/mass))
    print('Combined energy change (J/kg):',float((after['total']-before['total'])/mass))
    print('Minimum specific TKE (m²/s²):',float(state.specific_tke().min()))
    # model.step_with_report(state, dt) includes the existing dynamics followed
    # by vertical mixing. Its dynamics energy drift is a separate ledger.


if __name__=='__main__':main()
