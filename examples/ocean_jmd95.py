"""Active potential-temperature/salinity JMD95 ocean, explicit opt-in.

Small adiabatic CPU float64 demonstration; centered transport with rejected
invalid stages. This is not a bounded/mixed/forced climate configuration.
"""
import torch
from vfs.geometry.mesh import VoronoiMesh
from vfs.ocean import Ocean, OceanColumn
from vfs.ocean.jmd95 import JMD95Dynamics, checked_ssprk3


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    mesh=VoronoiMesh(n_cells=32,lloyd_iters=2,verbose=False)
    column=OceanColumn(3,total_depth=3000.)
    model=Ocean(mesh,column,dynamics=JMD95Dynamics(),stepper=checked_ssprk3)
    h=torch.tensor([500.,1000.,1500.]).expand(mesh.nCells,3).clone()
    temperature=torch.tensor([17.,10.,4.])[None,:]+.4*mesh.cell_xyz[:,:1]
    salinity=torch.tensor([34.,35.,35.5])[None,:]+.05*mesh.cell_xyz[:,1:2]
    state=model.state(u_e=torch.zeros(mesh.nEdges,3),h=h,temperature=temperature,salinity=salinity)
    initial=model.budgets(state)
    with torch.no_grad():
        for _ in range(10):
            state=model.step(state,30.)
    final=model.budgets(state)
    print('density range (kg/m3):',tuple(float(f(model.physical_fields(state)['density'])) for f in (torch.min,torch.max)))
    print('max speed (m/s):',float(state.u_e.abs().max()))
    for name in initial:
        print(name,'change:',float(final[name]-initial[name]) if initial[name] is not None else None)


if __name__=='__main__':
    main()
