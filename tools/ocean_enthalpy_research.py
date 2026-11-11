"""Research enthalpy-coordinate JMD95 ocean; not an installed runtime stepper.

Surface calorics are an explicit input. No atmosphere import or forward AD.
The bounded flux ledger is reused through a visibly named transport-only view;
that view must never be passed to a density or temperature diagnostic.
"""
from dataclasses import dataclass, replace
import math

import torch
from vfs.ocean.state import OceanState
from vfs.ocean.transport import HorizontalTransport, divergence


@dataclass(frozen=True)
class EnthalpyState:
    u_e: torch.Tensor
    h: torch.Tensor
    enthalpy_h: torch.Tensor
    salt_h: torch.Tensor
    tracer_h: object
    mesh: object
    column: object

    @classmethod
    def from_temperature(cls, state, reference):
        a = reference.evaluate(state.salinity(), state.temperature())['h0']
        return cls(state.u_e, state.h, state.h*a, state.salt_h, state.tracer_h,
                   state.mesh, state.column)

    def physical_state(self, reference):
        theta = reference.temperature_from_enthalpy(self.salt_h/self.h, self.enthalpy_h/self.h)
        return OceanState(self.u_e, self.h, self.h*theta, self.salt_h, self.tracer_h,
                          mesh=self.mesh, column=self.column)

    def transport_view(self):
        """Explicit generic-content adapter: first slot carries h*h0, NOT h*theta."""
        return OceanState(self.u_e, self.h, self.enthalpy_h, self.salt_h, self.tracer_h,
            mesh=self.mesh, column=self.column, attrs={'research_first_content': 'potential_enthalpy'})

    def __mul__(self, scalar):
        return replace(self, **{k: None if getattr(self, k) is None else scalar*getattr(self, k)
            for k in ('u_e', 'h', 'enthalpy_h', 'salt_h', 'tracer_h')})

    __rmul__ = __mul__

    def __add__(self, other):
        if self.mesh is not other.mesh or self.column is not other.column:
            raise ValueError('cannot combine states on different geometry')
        if (self.tracer_h is None) != (other.tracer_h is None):
            raise ValueError('cannot combine different tracer layouts')
        return replace(self, **{k: None if getattr(self, k) is None else getattr(self, k)+getattr(other, k)
            for k in ('u_e', 'h', 'enthalpy_h', 'salt_h', 'tracer_h')})


class EnthalpyEnergy:
    """H=H_JMD95(theta(h0,S),h,u)+rho0*sum(A*h*h0); eta from one surface G.

    JMD95 potential temperature is explicitly interpreted on IPTS-68 here.
    SA=SR is a reference-composition assumption, not a geographic conversion.
    All derivatives use fixed prognostic h, h*h0, h*SP as appropriate.
    """
    def __init__(self, base, reference):
        if reference.temperature_scale != 'IPTS-68':
            raise ValueError('this JMD95 research adapter requires explicit IPTS-68')
        self.base, self.reference = base, reference

    def mass_area(self, state):
        m = self.base.rho0*state.mesh.cell_area[:, None]
        return m if self.base.domain is None else m*self.base.domain.wet_cell[:, None]

    def fields(self, state, *, entropy_derivatives=False):
        physical = state.physical_state(self.reference)
        ref = self.reference.evaluate(physical.salinity(), physical.temperature(),
                                      entropy_derivatives=entropy_derivatives)
        return physical, ref

    def terms(self, state):
        result = self.base.terms(state.physical_state(self.reference))
        result['reference_thermal'] = (self.mass_area(state)*state.enthalpy_h).sum()
        result['total'] = result['total']+result['reference_thermal']
        return result

    def gradients(self, state, *, dynamic_only=False):
        physical, ref = self.fields(state)
        old = self.base.gradients(physical)
        a, s, theta = state.enthalpy_h/state.h, state.salt_h/state.h, physical.temperature()
        q = old['temp_h']/ref['h_theta']
        return dict(u_e=old['u_e'], h=old['h']+old['temp_h']*theta+q*(s*ref['h_s']-a),
                    enthalpy_h=q if dynamic_only else q+self.mass_area(state),
                    salt_h=old['salt_h']-q*ref['h_s'])

    def entropy(self, state):
        _, ref = self.fields(state)
        return (self.mass_area(state)*state.h*ref['entropy']).sum()

    def entropy_gradients(self, state):
        _, ref = self.fields(state, entropy_derivatives=True)
        ea = 1/ref['kelvin']
        es = ref['entropy_s']-ref['h_s']*ea
        m, a, s = self.mass_area(state), state.enthalpy_h/state.h, state.salt_h/state.h
        return dict(u_e=torch.zeros_like(state.u_e), h=m*(ref['entropy']-a*ea-s*es),
                    enthalpy_h=m*ea, salt_h=m*es)

    def thermodynamic_fields(self, state):
        g, e = self.gradients(state), self.entropy_gradients(state)
        wet = (self.mass_area(state) > 0).expand_as(state.h)
        m = torch.where(wet, self.mass_area(state), torch.ones_like(state.h))
        temperature = g['enthalpy_h']/torch.where(wet, e['enthalpy_h'], torch.ones_like(state.h))
        chemical = (g['salt_h']-temperature*e['salt_h'])/m
        if not torch.isfinite(temperature).all() or (temperature[wet] <= 0).any():
            raise FloatingPointError('nonpositive/nonfinite reduced thermodynamic temperature')
        # Dry cells participate in no graph edges; give them their surface T
        # for diagnostics, without dividing two masked zero derivatives.
        _, ref = self.fields(state)
        return torch.where(wet, temperature, ref['kelvin']), chemical


def transport_for(model, scheme='fct'):
    return HorizontalTransport(model.mesh, domain=model.domain, scheme=scheme,
                               min_thickness=model.dynamics.min_thickness)


def validate(model, state, energy):
    if energy.base is not model.dynamics.energy:
        raise ValueError('energy must borrow this model Hamiltonian')
    if model.sources:
        raise ValueError('research composition supports unforced dynamics only')
    physical = state.physical_state(energy.reference)
    model.check_state(physical)
    model.dynamics.check_state(physical)
    return physical


def momentum(model, state, energy):
    physical = validate(model, state, energy)
    g = energy.gradients(state, dynamic_only=True)
    m, mesh = model.dynamics.rho0*state.mesh.cell_area[:, None], state.mesh
    trace = model.dynamics.centered_trace
    pressure = -mesh.grad(g['h']/m)-trace(mesh, state.enthalpy_h/state.h)*mesh.grad(g['enthalpy_h']/m)
    pressure = pressure-trace(mesh, state.salt_h/state.h)*mesh.grad(g['salt_h']/m)
    if model.domain is not None:
        pressure = model.domain.apply_edge(pressure)
    return model.dynamics.mechanical.rotation_acceleration(physical)+pressure


@dataclass(frozen=True)
class WorkLedger:
    transport: object
    face_work: torch.Tensor
    thermal_power: torch.Tensor
    enthalpy_content_source: torch.Tensor


def evaluate(model, state, dt, energy, scheme='fct'):
    """Bounded enthalpy/salt plus equal-endpoint signed work conversion.

    This closes the instantaneous first law, NOT the second law. The centered
    force is derived in the new coordinates; reusing the old theta force would
    fail the adjoint pairing. No velocity division or sign-dependent force.
    """
    velocity_rate = momentum(model, state, energy)
    ledger = transport_for(model, scheme).advance(state.transport_view(), dt)
    area, nodes = state.mesh.cell_area[:, None], state.h.numel()
    fields = divergence(ledger.i, ledger.j, ledger.integrated_flux/dt, nodes).reshape(*state.h.shape, -1)/area[..., None]
    h_rate = divergence(ledger.i, ledger.j, ledger.carrier, nodes).reshape_as(state.h)/area
    g = energy.gradients(state, dynamic_only=True)
    covector = torch.stack([g[k]/area for k in ('enthalpy_h', 'salt_h')], -1).reshape(-1, 2)
    flux_change = (ledger.integrated_flux-ledger.high_integrated_flux)[:, :2]/dt
    work = ((covector[ledger.j]-covector[ledger.i])*flux_change).sum(-1)
    power = state.h.new_zeros(nodes).index_add(0, ledger.i, -.5*work).index_add(0, ledger.j, -.5*work).reshape_as(state.h)
    coefficient = g['enthalpy_h']+energy.mass_area(state)
    wet = (energy.mass_area(state) > 0).expand_as(state.h)
    if (coefficient[wet] <= 0).any():
        raise FloatingPointError('nonpositive enthalpy energy derivative')
    source = power/torch.where(wet, coefficient, torch.ones_like(coefficient))
    rhs = replace(state, u_e=velocity_rate, h=h_rate, enthalpy_h=fields[..., 0]+source,
                  salt_h=fields[..., 1], tracer_h=None if state.tracer_h is None else fields[..., 2:])
    return rhs, WorkLedger(ledger, work, power, source)


def euler(model, state, dt, energy, scheme='fct'):
    rhs, work = evaluate(model, state, dt, energy, scheme)
    ledger, area = work.transport, state.mesh.cell_area[:, None]
    amounts = ledger.amounts/area[..., None]
    end = replace(state, u_e=state.u_e+dt*rhs.u_e, h=ledger.volume/area,
        enthalpy_h=amounts[..., 0]+dt*work.enthalpy_content_source,
        salt_h=amounts[..., 1], tracer_h=None if state.tracer_h is None else amounts[..., 2:])
    validate(model, end, energy)  # Every full stage, including caloric domain.
    return end, work


def step(model, state, dt, energy, scheme='fct'):
    first, l1 = euler(model, state, dt, energy, scheme)
    second, l2 = euler(model, first, dt, energy, scheme)
    middle = .75*state+.25*second
    third, l3 = euler(model, middle, dt, energy, scheme)
    end = (1/3)*state+(2/3)*third
    validate(model, end, energy)
    return end, (l1, l2, l3)


def entropy_terms(energy, state, work, dt):
    """Independent face-form accounting, including centered entropy error."""
    e, ledger = energy.entropy_gradients(state), work.transport
    area = state.mesh.cell_area[:, None]
    covectors = [(e[k]/area).flatten() for k in ('h', 'enthalpy_h', 'salt_h')]
    jumps = [x[ledger.j]-x[ledger.i] for x in covectors]
    advective = ledger.carrier*jumps[0]+sum(ledger.integrated_flux[:, k]/dt*jumps[k+1] for k in range(2))
    thermal = e['enthalpy_h']*work.enthalpy_content_source
    return advective, thermal


def thermodynamic_exchange(model, state, dt, energy, *, heat_conductance, salt_mobility):
    """Fixed-geometry energy/salt transfer, same H and eta as advection.

    Positive conductance K (W/K) and mobility M (kg² PSU² K/(J s)) give face
    production K*dT²/(Ti*Tj)+alpha*M*d(mu/T)². Only a spatial statement;
    neither finite-step entropy nor joint salinity/enthalpy bounds follow.
    """
    validate(model, state, energy)
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if any(not math.isfinite(x) or x < 0 for x in (heat_conductance, salt_mobility)):
        raise ValueError('exchange coefficients must be finite and nonnegative')
    temperature, chemical = energy.thermodynamic_fields(state)
    _, i, j, _ = transport_for(model).graph(state.transport_view())
    t, mu = temperature.flatten(), chemical.flatten()
    energy_flux = heat_conductance*(t[i]-t[j])
    jump = mu[j]/t[j]-mu[i]/t[i]
    salt_flux = -salt_mobility*jump
    mass = model.dynamics.rho0*state.mesh.cell_area[:, None]
    amount = (mass*state.salt_h).flatten()
    out = torch.zeros_like(amount).index_add(0, i, dt*salt_flux.clamp_min(0)).index_add(0, j, dt*(-salt_flux).clamp_min(0))
    ratio = torch.where(out > 0, .9*amount/torch.where(out > 0, out, torch.ones_like(out)), torch.ones_like(out)).clamp(max=1)
    alpha = torch.where(salt_flux >= 0, ratio[i], ratio[j])
    salt_flux = alpha*salt_flux
    salt_rate = divergence(i, j, salt_flux, state.h.numel()).reshape_as(state.h)/mass
    energy_rate = divergence(i, j, energy_flux, state.h.numel()).reshape_as(state.h)
    g = energy.gradients(state)
    wet = (energy.mass_area(state) > 0).expand_as(state.h)
    q_rate = (energy_rate-g['salt_h']*salt_rate)/torch.where(wet, g['enthalpy_h'], torch.ones_like(state.h))
    rhs = replace(0*state, enthalpy_h=q_rate, salt_h=salt_rate)
    return rhs, dict(energy_flux=energy_flux, salt_flux=salt_flux, salt_limiter=alpha,
        entropy_production=heat_conductance*(t[i]-t[j]).square()/(t[i]*t[j])+alpha*salt_mobility*jump.square())


def contraction(gradient, rhs):
    products = [gradient[k]*getattr(rhs, k) for k in ('u_e', 'h', 'enthalpy_h', 'salt_h')]
    return sum(x.sum() for x in products), sum(x.abs().sum() for x in products)
