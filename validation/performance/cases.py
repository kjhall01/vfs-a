"""Explicit workloads; no dynamics equations or package defaults are changed."""
import sys
from pathlib import Path
import torch


def scalar(value):
    number = float(value)
    if not torch.isfinite(torch.tensor(number, dtype=torch.float64)):
        raise FloatingPointError('nonfinite endpoint diagnostic')
    return number


def report_metrics(reports):
    result = dict(external_energy_J=0., remap_energy_change_J=0.)
    for report, remap in reports:
        if remap is not None:
            result['remap_energy_change_J'] += scalar(remap.total_energy_change)
        for name,field in [('boundary','surface_work'),('heating','external_heat'),('exchange','external_energy')]:
            component = getattr(report,name,None)
            if component is not None: result['external_energy_J'] += scalar(getattr(component,field))
        mixing = getattr(report,'turbulence',report)
        if mixing is not None:
            if hasattr(mixing,'budget_error'):
                result['max_local_budget_ratio'] = max(result.get('max_local_budget_ratio',0.),scalar(mixing.budget_error))
            for key,value in (getattr(mixing,'mixing_limiter',None) or {}).items():
                key = 'mixing_'+key
                if key.endswith('minimum_fraction'): result[key] = min(result.get(key,1.),scalar(value))
                else: result[key] = result.get(key,0.)+scalar(value)
    return result


def make_case(c):
    from vfs import VoronoiMesh, CharneyPhillipsColumn, make_hybrid_coefficients, DryAtmosphere, MoistAtmosphere, State
    case, layers = c['case'],c['layers']
    if case in ('dry','moist'):
        mesh = VoronoiMesh(n_cells=c['cells'],lloyd_iters=0,rotation_scheme='compatible',verbose=False)
        a,b = make_hybrid_coefficients(layers,p_top=5000.)
        column = CharneyPhillipsColumn(layers,a,b,phi_full_mode='direct_u_gl2_adjoint')
        xyz = mesh.edge_xyz
        east = torch.linalg.cross(xyz.new_tensor([0.,0.,1.]).expand_as(xyz),xyz)
        wind = 10*(east*mesh.edge_normal).sum(-1)[:,None].expand(mesh.nEdges,layers)
        if case=='dry':
            model = DryAtmosphere(mesh,column).to(dtype=torch.float64)
            state = State.from_theta(wind,mesh.cell_area.new_full((mesh.nCells,layers+1),300.),
                mesh.cell_area.new_full((mesh.nCells,1),100000.),mesh=mesh,column=column)
            advance = lambda s,i:(model(s,c['dt']),(None,None))
        else:
            model = MoistAtmosphere(mesh,column)
            pressure = model.interface_pressure(100000.)
            state = model.state_from_fields(temperature=280*(pressure/100000.)**.08,
                surface_pressure=100000.,normal_wind=wind,vapor=.01,liquid=.001,ice=.0001)
            advance = lambda s,i:(model.step(s,c['dt']),(None,None))
        def diagnose(s):
            d = model.diagnostics(s)
            out = {key:scalar(d[key]) for key in ('mass', 'boundary_total_energy' if case=='dry' else 'energy')}
            if case=='moist':
                fields=model.physical_fields(s)
                out.update({f'min_{k}':scalar(fields[k].min()) for k in ('vapor','liquid','ice')})
                out['species_mass_kg']=[scalar(x) for x in d['species']]
            return out
        return state,advance,diagnose

    # The same nonuniform physical profiles used in the mixing resolution study.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ocean_column'))
    from resolution_mesh import setup
    from lagged import SubcycledLaggedClosure
    from vfs.ocean.stratified import (TKEOcean,TKEClosure,OceanBoundaryStress,
        OceanSurfaceHeating,OceanSurfaceExchange,OceanSurfaceFluxes,AdaptiveThermalRemapper)
    ocean,resolved,k = setup(c['cells'],layers)
    ocean.warm_start = c.get('warm_start', False)
    if not c.get('prepared', True) and hasattr(ocean.base, '_stratified_transport'):
        del ocean.base._stratified_transport
    closure = None
    if c['closure']=='supported': closure=TKEClosure(2.)
    elif c['closure'] in ('semi','fast'):
        closure=SubcycledLaggedClosure(2.,substeps=c['mixing_substeps'],fast_thermodynamics=c['closure']=='fast')
    full=case in ('full','full_ale')
    if case=='full_ale': ocean.remapper=AdaptiveThermalRemapper()
    model=TKEOcean(ocean,closure=closure,
        boundary=OceanBoundaryStress(.0025) if full else None,
        heating=OceanSurfaceHeating(.6,(.5,20.)) if full else None,
        exchange=OceanSurfaceExchange() if full else None)
    initial=model.state(resolved,tke=k)
    forcing={}
    if full:
        cell=lambda x:torch.full_like(resolved.h[:,0],x)
        xyz=ocean.mesh.edge_xyz
        east=torch.linalg.cross(xyz.new_tensor([0.,0.,1.]).expand_as(xyz),xyz)
        forcing=dict(surface_stress=.05*(east*ocean.mesh.edge_normal).sum(-1),
            shortwave=cell(200.),surface_heat=cell(-50.),surface_fluxes=OceanSurfaceFluxes(
                precipitation=cell(.00001),evaporation=cell(.000005),sensible_heat=cell(-10.),
                latent_heat=cell(-12.5),water_energy_flux=cell(.25),
                water_momentum_flux=torch.zeros_like(resolved.u_e[:,0])))
    def advance(state,index):
        if case=='mixing': out,report=model.mix_with_report(state,c['dt'])
        else: out,report=model.step_with_report(state,c['dt'],**forcing)
        remap=None
        if case=='full_ale' and (index+1)%c['remap_every']==0:
            # Explicit post-physics remap to equal fractions, preserving CURRENT depth.
            target=out.ocean.h.sum(-1,keepdim=True).expand_as(out.ocean.h)/layers
            out,remap=model.remap_with_report(out,target)
        return out,(report,remap)
    def diagnose(s):
        b=model.budgets(s)
        out={key:scalar(b[key]) for key in ('volume','salt','total','tke')}
        out['min_tke']=scalar(s.specific_tke().min())
        out['min_thickness']=scalar(s.ocean.h.min())
        out['min_tracer']=scalar((s.ocean.tracer_h/s.ocean.h[...,None]).min())
        out['passive_inventory']=scalar((ocean.energy.mass_area(s.ocean)[...,None]*s.ocean.tracer_h).sum())
        out['initial_mass_kg']=scalar((ocean.energy.mass_area(initial.ocean)*initial.ocean.h).sum())
        if out['min_tke']<0 or out['min_tracer']<0 or out['min_thickness']<=0:
            raise FloatingPointError('inadmissible endpoint')
        return out
    return initial,advance,diagnose
