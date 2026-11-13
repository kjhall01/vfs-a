"""Optional one-equation ocean turbulence with finite-step energy transfers.

k is specific TKE [m²/s²], stored as h*k on cells. Mixing length is prescribed,
not a second prognostic equation. Closed top/bottom; no wind/wave source.
No forward autograd, clipping, energy floors or alteration of dynamics kernels.
"""
from dataclasses import dataclass, replace
import math
from .heating import thermal_parts, apply_heat
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
    """Integrated transfers in J; entropy in J/K.

    buoyancy_work retains the column sum. buoyancy_interface_work and
    buoyancy_energy_flux have nlayer-1 entries (flux positive upward);
    buoyancy_layer_work/residual have nlayer entries.
    """
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
    buoyancy_interface_work: torch.Tensor
    buoyancy_layer_work: torch.Tensor
    buoyancy_energy_flux: torch.Tensor
    buoyancy_layer_residual: torch.Tensor


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

    _thermal_parts = staticmethod(thermal_parts)
    _heat = staticmethod(apply_heat)

    def _mix_adjacent(self, ocean, state, reservoir, conductance, dt):
        """Local implicit pair exchanges: even half, odd full, even half step.

        Each sweep contains disjoint layer pairs. For each pair the actual
        thermal-energy change W is shared by adjacent half-layer thicknesses.
        F=delta E_upper-alpha*W is the remaining upward energy transfer,
        where alpha=h_upper/(h_upper+h_lower). W is not
        confused with the much larger heat transported between layers.

        No nonadjacent TKE can pay a pair's cost, even if the column total is
        sufficient. Unaffordable exchanges are reduced before committing their paired flux.
        Frozen diffusivities and pairwise BE keep overall first-order accuracy;
        symmetric sweep ordering does not make the BE method second order.
        """
        h, energy = state.h, ocean.energy
        mass_area = energy.mass_area(state)
        mass = mass_area*h
        interface_work = torch.zeros_like(conductance)
        layer_work = torch.zeros_like(h)
        energy_flux = torch.zeros_like(conductance)
        layer_residual = torch.zeros_like(h)
        parity = torch.arange(conductance.shape[-1], device=h.device) % 2
        zero = torch.zeros_like(h[:, :1])
        divergence = lambda f: torch.cat((f, zero), -1)-torch.cat((zero, f), -1)
        adjacent_sum = lambda f: torch.cat((f, zero), -1)+torch.cat((zero, f), -1)
        for color, fraction in ((0, .5), (1, 1.), (0, .5)):
            g = torch.where(parity == color, conductance, torch.zeros_like(conductance))
            active = g > 0
            if not active.any():
                continue
            before = ocean.check_state(state)
            old_psi, old_ref = self._thermal_parts(ocean, before)
            salt0, enthalpy0 = before.salinity(), old_ref['h0']
            coefficient = fraction*dt*g/(1+fraction*dt*g*(1/h[:, :-1]+1/h[:, 1:]))
            def candidate(coefficient):
                def mix(q):
                    return q+divergence(coefficient*(q[:, 1:]-q[:, :-1]))/h
                salt_h = state.salt_h+divergence(coefficient*(salt0[:, 1:]-salt0[:, :-1]))
                salt, enthalpy = salt_h/h, mix(enthalpy0)
                nodes = adjacent_sum(active.to(h.dtype)) > 0
                theta = energy.reference.temperature_from_enthalpy(salt, enthalpy)
                eta = energy.reference.evaluate(salt, theta, entropy_derivatives=True)['entropy']
                # Subtract the same inverse/evaluate round trip at zero increment.
                # Uniform fields then remain bitwise unchanged WITHOUT branching on
                # their gradients (which would incorrectly erase the linearized
                # diffusion response at a uniform state).
                theta0 = energy.reference.temperature_from_enthalpy(salt0, enthalpy0)
                eta0 = energy.reference.evaluate(salt0, theta0, entropy_derivatives=True)['entropy']
                tracer = state.tracer_h
                if tracer is not None and tracer.shape[-1] > 0:
                    tracer = torch.stack([tracer[..., j]+divergence(coefficient*
                        ((tracer[..., j]/h)[:, 1:]-(tracer[..., j]/h)[:, :-1]))
                        for j in range(tracer.shape[-1])], -1)
                updated = replace(state, salt_h=salt_h,
                    entropy_h=torch.where(nodes, state.entropy_h+h*(eta-eta0), state.entropy_h), tracer_h=tracer)
                after = ocean.check_state(updated)
                new_psi, new_ref = self._thermal_parts(ocean, after)
                change = mass*((new_psi-old_psi)+(new_ref['h0']-old_ref['h0']))
                work = torch.where(active, change[:, :-1]+change[:, 1:], torch.zeros_like(g))
                alpha = h[:, :-1]/(h[:, :-1]+h[:, 1:])
                upper_work, lower_work = alpha*work, (1-alpha)*work
                debit = torch.cat((upper_work, zero), -1)+torch.cat((zero, lower_work), -1)
                return updated, change, work, upper_work, debit
            updated, change, work, upper_work, debit = candidate(coefficient)
            # Disjoint pairs compete only with their own two reservoirs. Each
            # subsequent sweep sees the remaining energy. Reducing the exchange preserves
            # convex pair mixing and never edits a negative reservoir afterward.
            for _ in range(48):
                bad_layer = debit > reservoir
                bad_pair = active & (bad_layer[:, :-1] | bad_layer[:, 1:])
                if not bad_pair.any():
                    break
                alpha = h[:, :-1]/(h[:, :-1]+h[:, 1:])
                budget = torch.minimum(reservoir[:, :-1]/alpha,
                                       reservoir[:, 1:]/(1-alpha))
                # Work need not be linear in exchanged content: this ratio is
                # only a trial reduction, always rechecked against the full H.
                positive_work = torch.where(work > 0, work, torch.ones_like(work))
                reduction = .5*(budget/positive_work).clamp(min=0, max=1)
                coefficient = torch.where(bad_pair, coefficient*reduction, coefficient)
                updated, change, work, upper_work, debit = candidate(coefficient)
            else:
                # Exact no-exchange fallback avoids unresolved thermodynamic
                # roundoff at vanishing flux; no heat or TKE floor is inserted.
                coefficient = torch.where(bad_pair, torch.zeros_like(coefficient), coefficient)
                updated, change, work, upper_work, debit = candidate(coefficient)
            if (debit > reservoir).any():
                raise ValueError('local buoyancy mixing exceeds available TKE in an adjacent layer; no nonlocal borrowing')
            flux = torch.where(active, change[:, :-1]-upper_work, torch.zeros_like(g))
            entropy_delta = mass_area*(updated.entropy_h-state.entropy_h)
            entropy_pair = entropy_delta[:, :-1]+entropy_delta[:, 1:]
            entropy_scale = mass_area*state.entropy_h.abs()
            entropy_tolerance = 512*torch.finfo(h.dtype).eps*(entropy_scale[:, :-1]+entropy_scale[:, 1:]).clamp_min(1)
            if (active & (entropy_pair < -entropy_tolerance)).any():
                raise ValueError('local scalar mixing decreases pair entropy')
            residual = change-debit-divergence(flux)
            reservoir = reservoir-debit
            state = updated
            interface_work = interface_work+work
            layer_work = layer_work+debit
            energy_flux = energy_flux+flux
            layer_residual = layer_residual+residual
        return state, reservoir, interface_work, layer_work, energy_flux, layer_residual

    def apply(self, ocean, state, tke_h, dt):
        """One split vertical-physics step. Inputs are never modified.

        Scalar mixing conserves potential enthalpy and salt, then debits the
        actual pair thermodynamic-energy change from adjacent layer TKE reservoirs.
        Local scalar exchange is limited to the adjacent available TKE.
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
        interface_work = torch.zeros_like(conductance)
        layer_work = torch.zeros_like(h)
        energy_flux = torch.zeros_like(conductance)
        layer_residual = torch.zeros_like(h)
        if self.mix_scalars and torch.any(conductance != 0):
            mixed, reservoir, interface_work, layer_work, energy_flux, layer_residual = self._mix_adjacent(
                ocean, mixed, reservoir, conductance/self.prandtl, dt)
            buoyancy = interface_work.sum(-1)
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
            kinetic_residual, cell_mass*thermal_residual, diffusion_residual, entropy_change, column_residual, column_exchange, global_residual,
            interface_work, layer_work, energy_flux, layer_residual)
