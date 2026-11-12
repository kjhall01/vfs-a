"""Optional one-equation ocean turbulence with finite-step energy transfers.

k is specific TKE [m²/s²], stored as h*k on cells. Mixing length is prescribed,
not a second prognostic equation. Closed top/bottom; no wind/wave source.
No forward autograd, clipping, energy floors or alteration of dynamics kernels.
"""
from dataclasses import dataclass, replace
import math
import torch


def positive_sqrt(x):
    """Zero value and zero branch derivative at k=0, without sqrt(0) in AD."""
    positive = x > 0
    return torch.where(positive, torch.sqrt(torch.where(positive, x, torch.ones_like(x))), torch.zeros_like(x))


def diffuse(value, mass, conductance, dt):
    """Closed-column backward Euler: M (x'-x) = -dt D^T G D x'.

    mass [column, layer], conductance [column, layer-1]. The symmetric
    tridiagonal matrix is positive definite for positive mass and G>=0.
    Works on scalar fields; no dense matrix or cross-column mixing.
    """
    if value.shape[-1] == 1 or not torch.any(conductance != 0):
        return value
    zero = torch.zeros_like(mass[:, :1])
    off = -dt*conductance
    diagonal = mass + torch.cat((dt*conductance, zero), -1) + torch.cat((zero, dt*conductance), -1)
    cp, yp = [], []
    flux = dt*conductance*(value[:, 1:]-value[:, :-1])
    rhs = torch.cat((flux, zero), -1)-torch.cat((zero, flux), -1)
    for j in range(value.shape[-1]):
        denom = diagonal[:, j] if j == 0 else diagonal[:, j]-off[:, j-1]*cp[-1]
        y = rhs[:, j] if j == 0 else rhs[:, j]-off[:, j-1]*yp[-1]
        yp.append(y/denom)
        if j < value.shape[-1]-1:
            cp.append(off[:, j]/denom)
    out = [yp[-1]]
    for j in range(value.shape[-1]-2, -1, -1):
        out.append(yp[j]-cp[j]*out[-1])
    return value+torch.stack(out[::-1], -1)


@dataclass(frozen=True)
class TKEReport:
    """All transfer arrays are J per cell/layer except column buoyancy [J]."""
    shear_production: torch.Tensor
    implicit_momentum_loss: torch.Tensor
    buoyancy_work: torch.Tensor
    dissipated_heat: torch.Tensor
    kinetic_residual: torch.Tensor
    thermal_residual: torch.Tensor
    tke_inventory_residual: torch.Tensor
    entropy_change: torch.Tensor
    column_energy_residual: torch.Tensor
    kinetic_column_exchange: torch.Tensor
    global_energy_residual: torch.Tensor


@dataclass(frozen=True)
class TKEClosure:
    """Experimental mixing-length closure; constants are explicit, not calibrated.

    K_m=C_mu L sqrt(k), K_h=K_m/Pr, K_k=K_m/sigma_k,
    epsilon=C_epsilon k^(3/2)/L. Set c_mu=0 for decay-only tests;
    mix_scalars=False isolates momentum/TKE exchange without buoyancy mixing.
    """
    mixing_length: float
    c_mu: float = 0.1
    c_epsilon: float = 0.7
    prandtl: float = 1.0
    sigma_k: float = 1.0
    mix_scalars: bool = True

    def __post_init__(self):
        for name in ('mixing_length', 'prandtl', 'sigma_k'):
            x = getattr(self, name)
            if not math.isfinite(x) or x <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('c_mu', 'c_epsilon'):
            x = getattr(self, name)
            if not math.isfinite(x) or x < 0:
                raise ValueError(f'{name} must be finite and nonnegative')

    @staticmethod
    def _thermal_parts(ocean, physical):
        energy = ocean.energy
        node, weight = energy.base._quadrature(physical.h)
        z = energy.base.height(physical)[..., None]+physical.h[..., None]*(node-.5)
        psi = energy.base.parcel_potential(physical.temperature()[..., None], physical.salinity()[..., None], z)[0]
        ref = energy.reference.evaluate(physical.salinity(), physical.temperature())
        return (psi*weight).sum(-1), ref

    @staticmethod
    def _heat(ocean, state, heat_specific):
        """Invert the full layer thermal energy increment at fixed h, SP and u.

        Integrate its analytic temperature derivative along the increment to
        avoid subtracting large reference thermal energies for tiny heating.
        Three-point Gauss quadrature evaluates the temperature-path integral;
        this local integral approximation is separately checked against H.
        """
        if not torch.any(heat_specific != 0):
            return state, torch.zeros_like(heat_specific)
        energy = ocean.energy; base = energy.base; reference = energy.reference
        physical = state.physical_state(reference)
        theta0, salt = physical.temperature(), physical.salinity()
        node, weight = base._quadrature(state.h)
        z = base.height(physical)[..., None]+state.h[..., None]*(node-.5)
        def derivative(theta):
            ref = reference.evaluate(salt, theta)
            pt = base.parcel_potential(theta[..., None], salt[..., None], z)[1]
            return ref['h_theta']+(pt*weight).sum(-1)
        nodes = (.5-math.sqrt(15)/10, .5, .5+math.sqrt(15)/10)
        weights = (5/18, 4/9, 5/18)
        delta = heat_specific/derivative(theta0)
        for _ in range(6):
            gained = delta*sum(w*derivative(theta0+t*delta) for t, w in zip(nodes, weights))
            delta = delta+(heat_specific-gained)/derivative(theta0+delta)
        gained = delta*sum(w*derivative(theta0+t*delta) for t, w in zip(nodes, weights))
        residual = gained-heat_specific
        if not torch.isfinite(delta).all() or (delta < 0).any():
            raise FloatingPointError('TKE heating inversion failed; no thermal clipping')
        theta = theta0+delta
        eta = reference.evaluate(salt, theta, entropy_derivatives=True)['entropy']
        updated = replace(state, entropy_h=torch.where(heat_specific != 0, state.h*eta, state.entropy_h))
        ocean.check_state(updated)
        return updated, residual

    def apply(self, ocean, state, tke_h, dt):
        """One split vertical-physics step. Inputs are never modified.

        Scalar mixing conserves potential enthalpy and salt, then debits the
        actual full thermodynamic-energy change from the column TKE reservoir.
        A step that exhausts TKE rejects; callers must reduce dt explicitly.
        """
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError('dt must be finite and positive')
        physical = ocean.check_state(state)
        h, mesh, energy = state.h, state.mesh, ocean.energy
        if tke_h.shape != h.shape or tke_h.dtype != h.dtype or tke_h.device != h.device:
            raise ValueError('tke_h must match thickness shape, dtype and device')
        if not torch.isfinite(tke_h).all() or (tke_h < 0).any():
            raise ValueError('TKE must be finite and nonnegative')
        mass_area = energy.mass_area(state)
        wet = (mass_area > 0).expand_as(h)
        if (tke_h[~wet] != 0).any():
            raise ValueError('dry-cell TKE must be zero')
        safe_area = torch.where(wet, mass_area, torch.ones_like(h))
        cell_mass = mass_area*h
        k = tke_h/h
        diffusivity = self.c_mu*self.mixing_length*positive_sqrt(k)
        dz = .5*(h[:, :-1]+h[:, 1:])
        face_k = .5*(diffusivity[:, :-1]+diffusivity[:, 1:])
        conductance = face_k/dz
        # Native edge metric is the exact Hessian of the resolved kinetic H.
        mechanical = energy.base.mechanical
        he = mesh.cell_to_edge(h)
        metric = mechanical.metric(physical)
        edge_k = mesh.cell_to_edge(diffusivity)
        edge_g = ocean.column.rho0*(mesh.dc_edge*mesh.dv_edge)[:, None] * \
            (.5*(edge_k[:, :-1]+edge_k[:, 1:]))/(.5*(he[:, :-1]+he[:, 1:]))
        if ocean.domain is not None:
            edge_g = edge_g*ocean.domain.active_edge[:, None]
        u = diffuse(state.u_e, metric, edge_g, dt)
        # BE energy identity: physical shear loss plus finite-step damping.
        exchange = dt*edge_g*(u[:, 1:]-u[:, :-1]).square()
        zero = torch.zeros_like(u[:, :1])
        edge_shear = .5*(torch.cat((exchange, zero), -1)+torch.cat((zero, exchange), -1))
        edge_time = .5*metric*(u-state.u_e).square()
        # Same cell allocation as K_c = h_c (dK/dh)_c, including unequal
        # neighbouring thicknesses: partition weights are w_ce*h_c/I(h)_e.
        allocate = lambda edge_energy: h*mechanical.interpolation_transpose(mesh, edge_energy/he)
        shear = allocate(edge_shear)
        time_loss = allocate(edge_time)
        kinetic_change = (.5*metric*(u-state.u_e)*(u+state.u_e)).sum()
        kinetic_residual = kinetic_change+shear.sum()+time_loss.sum()
        reservoir = mass_area*tke_h+shear+time_loss
        mixed = replace(state, u_e=u)
        buoyancy = h.new_zeros(h.shape[0])
        if self.mix_scalars and torch.any(conductance != 0):
            old_psi, old_ref = self._thermal_parts(ocean, physical)
            salt = diffuse(physical.salinity(), h, conductance/self.prandtl, dt)
            enthalpy = diffuse(old_ref['h0'], h, conductance/self.prandtl, dt)
            theta = energy.reference.temperature_from_enthalpy(salt, enthalpy)
            eta = energy.reference.evaluate(salt, theta, entropy_derivatives=True)['entropy']
            tracer = state.tracer_h
            if tracer is not None and tracer.shape[-1] > 0:
                tracer = torch.stack([h*diffuse(tracer[..., j]/h, h, conductance/self.prandtl, dt)
                                      for j in range(tracer.shape[-1])], -1)
            mixed = replace(mixed, salt_h=torch.where(wet, h*salt, state.salt_h),
                entropy_h=torch.where(wet, h*eta, state.entropy_h), tracer_h=tracer)
            mixed_physical = ocean.check_state(mixed)
            new_psi, new_ref = self._thermal_parts(ocean, mixed_physical)
            # Include caloric inversion/solver residual, not just an N² proxy.
            buoyancy = (cell_mass*((new_psi-old_psi)+(new_ref['h0']-old_ref['h0']))).sum(-1)
            available = reservoir.sum(-1)
            if (buoyancy > available).any():
                raise ValueError('buoyancy mixing exceeds available TKE; reduce dt')
            # Positive cost draws proportionally from available layer TKE;
            # released energy is distributed by layer mass, locally per column.
            safe = torch.where(available > 0, available, torch.ones_like(available))
            fractions = reservoir/safe[:, None]
            mass_fraction = h/h.sum(-1, keepdim=True)
            reservoir = reservoir-torch.where((buoyancy > 0)[:, None],
                fractions*buoyancy[:, None], mass_fraction*buoyancy[:, None])
        # Closed vertical transport of TKE, then exact local decay for fixed L.
        before_diffusion = reservoir.sum()
        kt = diffuse(reservoir/safe_area/h, h, conductance/self.sigma_k, dt)
        kt = torch.where(wet, kt, torch.zeros_like(kt))
        diffusion_residual = (cell_mass*kt).sum()-before_diffusion
        if not torch.isfinite(kt).all() or (kt < 0).any():
            raise FloatingPointError('TKE diffusion failed positivity; reduce dt')
        end_k = kt/(1+.5*dt*self.c_epsilon*positive_sqrt(kt)/self.mixing_length).square()
        heat_specific = kt-end_k
        end, thermal_residual = self._heat(ocean, mixed, heat_specific)
        end_tke = h*end_k
        entropy_change = (mass_area*(end.entropy_h-state.entropy_h)).sum(-1)
        tolerance = 512*torch.finfo(h.dtype).eps*(mass_area*state.entropy_h.abs()).sum(-1).clamp_min(1)
        if not torch.isfinite(end_tke).all() or (end_tke < 0).any():
            raise FloatingPointError('TKE positivity failed; no floor')
        if (entropy_change < -tolerance).any():
            raise ValueError('TKE step decreases column entropy; reduce dt or revise mixing state')
        final_physical = ocean.check_state(end)
        old_psi, old_ref = self._thermal_parts(ocean, physical)
        new_psi, new_ref = self._thermal_parts(ocean, final_physical)
        thermal_change = cell_mass*((new_psi-old_psi)+(new_ref['h0']-old_ref['h0']))
        local_kinetic = allocate(.5*metric*(u-state.u_e)*(u+state.u_e))
        raw_column_change = (thermal_change+local_kinetic+mass_area*(end_tke-tke_h)).sum(-1)
        # When h varies horizontally, the cell fraction of an edge's kinetic
        # energy varies with layer. Vertical edge work then redistributes the
        # cell-assigned K between neighbours. This known zero-global-sum term
        # belongs in the local budget; it is not extra TKE or thermal heating.
        column_exchange = (local_kinetic+shear+time_loss).sum(-1)
        column_residual = raw_column_change-column_exchange
        global_residual = raw_column_change.sum()
        scale = (cell_mass*(old_psi.abs()+old_ref['h0'].abs())+
                 local_kinetic.abs()+mass_area*tke_h).sum(-1).clamp_min(1)
        tolerance = 1024*torch.finfo(h.dtype).eps*scale
        if (column_residual.abs() > tolerance).any() or global_residual.abs() > tolerance.sum():
            raise FloatingPointError('TKE full-H energy closure failed')
        return end, end_tke, TKEReport(shear, time_loss, buoyancy, cell_mass*heat_specific,
            kinetic_residual, cell_mass*thermal_residual, diffusion_residual, entropy_change, column_residual, column_exchange, global_residual)
