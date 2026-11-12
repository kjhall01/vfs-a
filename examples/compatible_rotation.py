"""Unperturbed dry JW initialization with compatible horizontal rotation."""
import torch
from vfs import VoronoiMesh, DryAtmosphere, make_hybrid_coefficients
from vfs.atmosphere.initial_conditions.jablonski_williamson import JWInitialConditions

mesh=VoronoiMesh(n_cells=256,lloyd_iters=5,rotation_scheme='compatible')
A,B=make_hybrid_coefficients(12,p_top=5000.)
column=JWInitialConditions.make_column_for_mode(mesh,12,A,B,
    mode='constant_ps_surface_phi',phi_full_mode='direct_u_gl2_adjoint')
model=DryAtmosphere(mesh,column)
state=JWInitialConditions(mesh,column,mode='constant_ps_surface_phi').make_state(
    model.dynamics,perturb=False,project_mass_flux=False)
with torch.no_grad():
    work=model.dynamics.rotation_work_terms(state)
    state=model.step(state,10.)
print('Rotation relative work:',float(work['work_relative']))
print('Maximum wind after 10 s (m/s):',float(state.u_e.abs().max()))
