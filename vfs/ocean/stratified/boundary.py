"""Prescribed surface stress and stationary-bed horizontal drag for the TKE ocean.

Stress is edge-normal force per horizontal area [Pa]. Native kinetic metric
weights turn it into an impulse/work ledger. No atmospheric bulk law, heat or
freshwater flux, sloping-bed projection or separate surface TKE flux is implied.
"""
from dataclasses import dataclass, replace
import math
import torch
from .tke import positive_sqrt


def wind_velocity(velocity, stress, thickness, rho0, dt):
    """Constant-stress kick at fixed carrier thickness; tensors may be batched."""
    return velocity+dt*stress/(rho0*thickness)


def drag_velocity(velocity, speed, thickness, coefficient, dt):
    """Backward Euler with nonnegative speed frozen at the substep entrance."""
    factor=dt*coefficient*speed/thickness
    if not torch.isfinite(factor).all():
        raise FloatingPointError('nonfinite bottom-drag factor')
    return velocity/(1+factor)


def bottom_speed(mesh, velocity):
    """Native normal component plus a reconstructed tangential component.

    Reconstruct cell Cartesian currents, interpolate them to the edge and
    project onto the local horizontal tangent orthogonal to the edge normal.
    This estimates the unresolved vector direction; energy closure uses the
    native edge metric, not the kinetic energy of the reconstruction.
    """
    cells=mesh.cell_wind_vector(velocity[:, None])[:, 0, :]
    reconstructed=mesh.cell_to_edge(cells)
    tangent=torch.linalg.cross(mesh.edge_xyz, mesh.edge_normal)
    tangential=(reconstructed*tangent).sum(-1)
    return positive_sqrt(velocity.square()+tangential.square())


@dataclass(frozen=True)
class BoundaryStressReport:
    """Impulses are native edge quantities [N s]; work/energy fields are J."""
    surface_impulse: torch.Tensor
    bottom_impulse: torch.Tensor
    surface_work_edge: torch.Tensor
    surface_work: torch.Tensor
    bottom_ke_loss: torch.Tensor
    bottom_tke_gain: torch.Tensor
    momentum_residual: torch.Tensor
    column_energy_residual: torch.Tensor
    energy_residual: torch.Tensor


@dataclass(frozen=True)
class OceanBoundaryStress:
    """Optional forcing with constant dimensionless Cd (zero disables drag).

    Prescribed wind stress acts first; horizontal bottom drag acts second.
    Bathymetry may vary. Stress is per horizontal area on the local bottom
    layer; there is no seabed-normal projection or sloping-area multiplier.
    All drag-related resolved KE loss becomes bottom-layer TKE. Boundary work
    is first-order split from dynamics/mixing. No surface TKE input is added.
    """
    bottom_drag_coefficient: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.bottom_drag_coefficient) or self.bottom_drag_coefficient < 0:
            raise ValueError('bottom_drag_coefficient must be finite and nonnegative')

    def _velocity_state(self,state,velocity):
        return replace(state,u_e=velocity)

    def apply(self, ocean, state, tke_h, dt, *, surface_stress=None, _physical=None, _return_physical=False):
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError('dt must be finite and positive')
        physical=(ocean.check_state(state) if _physical is None else ocean.check_state(state, _physical=_physical))
        h, mesh=state.h, state.mesh
        if tke_h.shape != h.shape or tke_h.dtype != h.dtype or tke_h.device != h.device:
            raise ValueError('tke_h must match thickness shape, dtype and device')
        if not torch.isfinite(tke_h).all() or (tke_h < 0).any():
            raise ValueError('TKE must be finite and nonnegative')
        mass_area=ocean.energy.mass_area(state)
        wet=(mass_area > 0).expand_as(h)
        if (tke_h[~wet] != 0).any():
            raise ValueError('dry-cell TKE must be zero')
        if surface_stress is None:
            stress=torch.zeros_like(state.u_e[:, 0])
        else:
            if not isinstance(surface_stress, torch.Tensor):
                raise TypeError('surface_stress must be an edge-normal tensor in Pa')
            if (surface_stress.shape != (mesh.nEdges,) or surface_stress.dtype != h.dtype or
                    surface_stress.device != h.device or not torch.isfinite(surface_stress).all()):
                raise ValueError('surface_stress must be finite [edge], matching state dtype/device')
            stress=surface_stress
        active=torch.ones(mesh.nEdges, dtype=torch.bool, device=h.device) if ocean.domain is None else ocean.domain.active_edge
        stress=torch.where(active, stress, torch.zeros_like(stress))
        cd=self.bottom_drag_coefficient
        if cd > 0:
            depth=ocean.column.total_depth_like(h)[:, 0]
            wet_depth=depth[wet[:, 0]]
            if ocean.domain is not None and ocean.domain.depth is not None:
                domain_depth=ocean.domain.depth[:, 0][wet[:, 0]]
                if not torch.equal(domain_depth, wet_depth):
                    raise ValueError('bottom drag requires consistent column/domain depth')
        energy=ocean.energy.base.mechanical
        metric=energy.metric(physical)
        he=mesh.cell_to_edge(h)
        area=mesh.dc_edge*mesh.dv_edge
        forced_top=wind_velocity(state.u_e[:, 0], stress, he[:, 0], ocean.column.rho0, dt)
        forced=torch.cat((forced_top[:, None],state.u_e[:, 1:]),-1)
        surface_impulse=dt*area*stress
        surface_work=surface_impulse*.5*(state.u_e[:, 0]+forced_top)
        speed=bottom_speed(mesh,forced[:, -1]) if cd > 0 else torch.zeros_like(forced[:, -1])
        speed=torch.where(active,speed,torch.zeros_like(speed))
        bottom=drag_velocity(forced[:, -1],speed,he[:, -1],cd,dt)
        velocity=torch.cat((forced[:, :-1],bottom[:, None]),-1)
        bottom_impulse=-dt*area*ocean.column.rho0*cd*speed*bottom
        loss=-.5*metric[:, -1]*(bottom-forced[:, -1])*(bottom+forced[:, -1])
        if not torch.isfinite(loss).all() or (loss < 0).any():
            raise FloatingPointError('bottom drag failed dissipativity')
        edge_loss=torch.cat((torch.zeros_like(state.u_e[:, :-1]),loss[:, None]),-1)
        allocate=lambda q: h*energy.interpolation_transpose(mesh,q/he)
        gain=allocate(edge_loss)
        safe_mass=torch.where(mass_area > 0,mass_area,torch.ones_like(mass_area))
        end_tke=tke_h+gain/safe_mass
        end=self._velocity_state(state,velocity)
        final=physical.with_tensors(end.u_e,physical.h,physical.temp_h,physical.salt_h,physical.tracer_h)
        final = (ocean.check_state(end) if _physical is None else
                 ocean.check_state(end, _physical=final))
        if not torch.isfinite(end_tke).all() or (end_tke < 0).any():
            raise FloatingPointError('boundary TKE update failed')
        kinetic_change=.5*metric*(velocity-state.u_e)*(velocity+state.u_e)
        wind_edge=torch.cat((surface_work[:, None],torch.zeros_like(state.u_e[:, 1:])), -1)
        column_residual=(allocate(kinetic_change-wind_edge)+mass_area*(end_tke-tke_h)).sum(-1)
        momentum_residual=(metric*(velocity-state.u_e)).sum(-1)-surface_impulse-bottom_impulse
        residual=column_residual.sum()
        scale=(allocate((.5*metric*state.u_e.square()).abs()+wind_edge.abs())+
               mass_area*tke_h+gain).sum(-1).clamp_min(1)
        tolerance=512*torch.finfo(h.dtype).eps*scale
        if (column_residual.abs() > tolerance).any():
            raise FloatingPointError('boundary work/TKE energy closure failed')
        result = (end,end_tke,BoundaryStressReport(surface_impulse,bottom_impulse,surface_work,
            surface_work.sum(),loss,gain,momentum_residual,column_residual,residual))
        return (*result,final) if _return_physical else result
