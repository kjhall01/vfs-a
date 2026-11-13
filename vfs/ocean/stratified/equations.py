"""Research Gibbs/entropy-coordinate ocean, reusing atmospheric design ideas.

Ocean-owned equations; no atmosphere import. The surface potential is derived
from GSW evaluations and JMD95 remains the density EOS. This candidate uses
strictly positive active salinity; zero passive tracers remain supported.
"""
from dataclasses import dataclass, replace
import torch

from vfs.ocean.state import OceanState
from vfs.ocean.transport import HorizontalTransport, divergence
from .faces import gibbs_face, temperature_from_entropy


@dataclass(frozen=True)
class EntropyState:
    u_e: torch.Tensor
    h: torch.Tensor
    entropy_h: torch.Tensor
    salt_h: torch.Tensor
    tracer_h: object
    mesh: object
    column: object

    @classmethod
    def from_temperature(cls, state, reference):
        eta = reference.evaluate(state.salinity(), state.temperature(), entropy_derivatives=True)['entropy']
        return cls(state.u_e, state.h, state.h*eta, state.salt_h, state.tracer_h, state.mesh, state.column)

    def physical_state(self, reference, *, guess=None):
        theta = temperature_from_entropy(reference, self.salt_h/self.h, self.entropy_h/self.h, guess=guess)
        return OceanState(self.u_e, self.h, self.h*theta, self.salt_h, self.tracer_h,
                          mesh=self.mesh, column=self.column)

    def transport_view(self):
        """Only for the generic content ledger; first slot is entropy, not T."""
        return OceanState(self.u_e, self.h, self.entropy_h, self.salt_h, self.tracer_h,
            mesh=self.mesh, column=self.column, attrs={'research_first_content': 'entropy'})

    def __mul__(self, scalar):
        return replace(self, **{k: None if getattr(self, k) is None else scalar*getattr(self, k)
            for k in ('u_e', 'h', 'entropy_h', 'salt_h', 'tracer_h')})

    __rmul__ = __mul__

    def __add__(self, other):
        if self.mesh is not other.mesh or self.column is not other.column:
            raise ValueError('cannot combine states on different geometry')
        if (self.tracer_h is None) != (other.tracer_h is None):
            raise ValueError('cannot combine different tracer layouts')
        return replace(self, **{k: None if getattr(self, k) is None else getattr(self, k)+getattr(other, k)
            for k in ('u_e', 'h', 'entropy_h', 'salt_h', 'tracer_h')})


class EntropyEnergy:
    """The same physical H as #35, differentiated at fixed h,h*eta,h*SP."""
    def __init__(self, base, reference):
        if reference.temperature_scale != 'IPTS-68':
            raise ValueError('JMD95 adapter requires explicit IPTS-68')
        self.base, self.reference = base, reference

    def mass_area(self, state):
        mass = self.base.rho0*state.mesh.cell_area[:, None]
        return mass if self.base.domain is None else mass*self.base.domain.wet_cell[:, None]

    def fields(self, state):
        physical = state.physical_state(self.reference)
        ref = self.reference.evaluate(physical.salinity(), physical.temperature(), entropy_derivatives=True)
        return physical, ref

    def terms(self, state):
        physical, ref = self.fields(state)
        result = self.base.terms(physical)
        result['reference_thermal'] = (self.mass_area(state)*state.h*ref['h0']).sum()
        result['total'] = result['total']+result['reference_thermal']
        return result

    def gradients(self, state, *, dynamic_only=False):
        physical, ref = self.fields(state)
        old = self.base.gradients(physical)
        return self._gradients(state, physical, ref, old, dynamic_only=dynamic_only)

    def _gradients(self, state, physical, ref, old, *, dynamic_only=False):
        """Apply the coordinate transform to already evaluated stage fields."""
        eta, salt = state.entropy_h/state.h, state.salt_h/state.h
        q = old['temp_h']/ref['entropy_theta']
        g = dict(u_e=old['u_e'], h=old['h']+old['temp_h']*physical.temperature()+q*(salt*ref['entropy_s']-eta),
                 entropy_h=q, salt_h=old['salt_h']-q*ref['entropy_s'])
        if not dynamic_only:
            temperature = ref['kelvin']
            mu = ref['h_s']-temperature*ref['entropy_s']
            m = self.mass_area(state)
            g['h'] = g['h']+m*(ref['h0']-temperature*eta-salt*mu)
            g['entropy_h'] = g['entropy_h']+m*temperature
            g['salt_h'] = g['salt_h']+m*mu
        return g

    def entropy(self, state):
        return (self.mass_area(state)*state.entropy_h).sum()


def validate(model, state, energy, *, physical=None, guess=None):
    """Validate fields; an internal caller may supply a residual-checked inversion.

    The supplied physical state must represent these exact entropy contents.
    Public model checks continue to reconstruct it independently by default.
    """
    if energy.base is not model.dynamics.energy or model.sources:
        raise ValueError('research composition requires the same unforced JMD95 model')
    salt = state.salt_h/state.h
    if ((salt <= 0)|(salt >= 42)).any():
        raise ValueError('entropy candidate requires 0 < SP < 42; no salt floor')
    if state.tracer_h is not None and (state.tracer_h < 0).any():
        raise ValueError('research passive tracers must be nonnegative')
    if physical is None:
        physical = state.physical_state(energy.reference, guess=guess)
    theta90 = physical.temperature()*energy.reference.to_90
    if ((theta90 <= -2.5)|(theta90 >= 40)).any():
        raise ValueError('entropy candidate requires interior reference temperatures')
    model.check_state(physical)
    model.dynamics.check_state(physical)
    return physical


def transport_for(model):
    if hasattr(model, '_stratified_transport'):
        return model._stratified_transport
    return HorizontalTransport(model.mesh, domain=model.domain, scheme='donor',
                               min_thickness=model.dynamics.min_thickness)


def high_traces(model, physical, energy):
    mesh = model.mesh
    i, j = mesh.cellsOnEdge.unbind(1)
    eta, salt, audit = gibbs_face(energy.reference, physical.salinity()[i], physical.temperature()[i],
                                physical.salinity()[j], physical.temperature()[j])
    return eta, salt, audit


def momentum(model, state, energy, physical, eta, salt, *, gradient=None):
    # The surface reference cancels by the Gibbs identity. Remove that zero
    # analytically; no subtraction of large caloric force components.
    g = energy.gradients(state, dynamic_only=True) if gradient is None else gradient
    mesh, mass = state.mesh, model.dynamics.rho0*state.mesh.cell_area[:, None]
    pressure = -mesh.grad(g['h']/mass)-eta*mesh.grad(g['entropy_h']/mass)-salt*mesh.grad(g['salt_h']/mass)
    if model.domain is not None:
        pressure = model.domain.apply_edge(pressure)
    return model.dynamics.mechanical.rotation_acceleration(physical)+pressure


@dataclass(frozen=True)
class AdmissibleLedger:
    volume: torch.Tensor
    amounts: torch.Tensor
    integrated_flux: torch.Tensor
    high_integrated_flux: torch.Tensor
    trace_change: torch.Tensor
    beta: torch.Tensor
    i: torch.Tensor
    j: torch.Tensor
    carrier: torch.Tensor
    outflow_fraction: torch.Tensor


def bounded_transport(model, state, dt, eta_high, salt_high, composition_entropy, *, low=None):
    """Gibbs high flux, donor reference, all-face salt/passive amount budgets.

    Entropy change is attached to the ACTUAL salt trace change using the
    endpoint-mean fixed-T composition derivative, as in the atmospheric design.
    It is not an independently upwinded entropy flux. Passives have independent
    betas and cannot trigger thermodynamic force corrections. Bounds are
    0<SP<42 and passive>=0, NOT local extrema. Check the full caloric domain.
    """
    if low is None:
        low = transport_for(model).advance(state.transport_view(), dt)
    active = slice(None) if model.domain is None else model.domain.active_edge
    scalars = [state.entropy_h[..., None], state.salt_h[..., None]]
    if state.tracer_h is not None:
        scalars.append(state.tracer_h)
    q = (torch.cat(scalars, -1)/state.h[..., None]).reshape(state.h.numel(), -1)
    high = torch.cat([eta_high[active].reshape(-1, 1), salt_high[active].reshape(-1, 1),
                      .5*(q[low.i, 2:]+q[low.j, 2:])], -1)
    donor = torch.where((low.carrier >= 0)[:, None], q[low.i], q[low.j])
    high_flux = (dt*low.carrier)[:, None]*high
    anti = high_flux-low.donor_integrated_flux
    baseline = low.low_amounts.reshape_as(q)
    salt_upper = 42*low.volume.flatten()
    def ratio(budget, demand):
        denominator = torch.where(demand > 0, demand, torch.ones_like(demand))
        return torch.where(demand > 0, budget.clamp_min(0)/denominator, torch.ones_like(demand)).clamp(max=1)
    # Allocate each node's capacity equally to ALL incident faces. Then each
    # individual face recovers beta=1 as its flux vanishes, even while OTHER
    # faces remain limited. A common node ratio would not ensure that limit.
    prepared = getattr(transport_for(model), '_prepared', None)
    if prepared is None:
        degree = torch.zeros(q.shape[0], dtype=q.dtype, device=q.device)
        degree = degree.index_add(0, low.i, torch.ones_like(low.carrier)).index_add(0, low.j, torch.ones_like(low.carrier))
    else:
        degree = prepared['degree']
    degree = degree.clamp_min(1)[:, None]  # isolated dry nodes have no faces
    lower_capacity = baseline[:, 1:]/degree
    # Reserve rounding room in the finite upper bound when limiting. Scaling
    # beta by epsilon alone cannot prevent cancellation from rounding SP to 42.
    # This reduces antidiffusive flux, never floors or clips a stored salinity.
    room = (salt_upper-baseline[:, 1]).clamp_min(0)
    upper_margin = torch.minimum(64*torch.finfo(q.dtype).eps*(salt_upper.abs()+baseline[:, 1].abs()), .5*room)
    upper_capacity = (room-upper_margin)/degree[:, 0]
    demand = anti[:, 1:].abs()
    losing = torch.where(anti[:, 1:] >= 0, lower_capacity[low.i], lower_capacity[low.j])
    beta = ratio(losing, demand)
    receiving = torch.where(anti[:, 1] >= 0, upper_capacity[low.j], upper_capacity[low.i])
    beta = torch.cat([torch.minimum(beta[:, 0], ratio(receiving, demand[:, 0]))[:, None], beta[:, 1:]], -1)
    beta = torch.where(beta < 1, beta*(1-64*torch.finfo(q.dtype).eps), beta)
    beta = torch.cat([beta[:, :1], beta], -1)
    amounts = baseline+divergence(low.i, low.j, beta*anti, q.shape[0])
    if not torch.isfinite(amounts).all() or (amounts[:, 1] <= 0).any() or (amounts[:, 1] >= salt_upper).any() or (amounts[:, 2:] < 0).any():
        raise FloatingPointError('admissible amount ledger failed; no clipping')
    flux = low.donor_integrated_flux+beta*anti
    trace_change = (1-beta)*(donor-high)
    delta_eta = composition_entropy[active].flatten()*trace_change[:, 1]
    trace_change = torch.cat([delta_eta[:, None], trace_change[:, 1:]], -1)
    entropy_flux = dt*low.carrier*(high[:, 0]+delta_eta)
    initial_entropy = (state.mesh.cell_area[:, None]*state.entropy_h).flatten()
    entropy_amount = initial_entropy+divergence(low.i, low.j, entropy_flux, q.shape[0])
    amounts = torch.cat([entropy_amount[:, None], amounts[:, 1:]], -1)
    flux = torch.cat([entropy_flux[:, None], flux[:, 1:]], -1)
    return AdmissibleLedger(low.volume, amounts.reshape(*state.h.shape, -1),
        flux, high_flux, trace_change, beta,
        low.i, low.j, low.carrier, low.outflow_fraction)


@dataclass(frozen=True)
class WorkLedger:
    transport: AdmissibleLedger
    face_work: torch.Tensor
    heat: torch.Tensor
    entropy_source: torch.Tensor
    force_correction: torch.Tensor
    gibbs_identity: torch.Tensor


class _StageEvaluation:
    """Work shared only within one RHS evaluation of one unchanged state.

    Nothing is attached to a model or state. New SSPRK stages get new work;
    tensor graphs are retained for differentiation. Validation and every
    original thermodynamic expression remain in force.
    """
    def __init__(self, model, state, dt, energy, *, physical=None):
        self.physical = validate(model, state, energy, physical=physical)
        self.eta, self.salt, self.face = high_traces(model, self.physical, energy)
        ref = energy.reference.evaluate(self.physical.salinity(),
            self.physical.temperature(), entropy_derivatives=True)
        old = energy.base.gradients(self.physical)
        self.dynamic_gradient = energy._gradients(state, self.physical, ref, old,
            dynamic_only=True)
        self.gradient = energy._gradients(state, self.physical, ref, old)
        self.low = transport_for(model).advance(state.transport_view(), dt)


def evaluate(model, state, dt, energy, *, _stage=None):
    stage = _StageEvaluation(model, state, dt, energy) if _stage is None else _stage
    physical, eta, salt, face = stage.physical, stage.eta, stage.salt, stage.face
    velocity = momentum(model, state, energy, physical, eta, salt,
                        gradient=stage.dynamic_gradient)
    ledger = bounded_transport(model, state, dt, eta, salt,
                               face['composition_entropy'], low=stage.low)
    area, nodes = state.mesh.cell_area[:, None], state.h.numel()
    g = stage.gradient
    covector = torch.stack([g[k]/area for k in ('entropy_h', 'salt_h')], -1).reshape(-1, 2)
    per_carrier = ((covector[ledger.j]-covector[ledger.i])*ledger.trace_change[:, :2]).sum(-1)
    work = ledger.carrier*per_carrier
    heat = (-work).clamp_min(0)
    mechanical = torch.where(work > 0, per_carrier, torch.zeros_like(per_carrier))
    coefficient = g['entropy_h'].flatten()
    wet = (energy.mass_area(state) > 0).expand_as(state.h).flatten()
    if not torch.isfinite(coefficient).all() or (coefficient[wet] <= 0).any():
        raise FloatingPointError('nonpositive entropy energy derivative')
    source = torch.zeros_like(coefficient).index_add(0, ledger.i, .5*heat/coefficient[ledger.i])
    source = source.index_add(0, ledger.j, .5*heat/coefficient[ledger.j]).reshape_as(state.h)
    active = slice(None) if model.domain is None else model.domain.active_edge
    correction = torch.zeros_like(state.u_e)
    correction[active] = -mechanical.reshape_as(correction[active])/(model.dynamics.rho0*model.mesh.dc_edge[active, None])
    rates = divergence(ledger.i, ledger.j, ledger.integrated_flux/dt, nodes).reshape(*state.h.shape, -1)/area[..., None]
    h_rate = divergence(ledger.i, ledger.j, ledger.carrier, nodes).reshape_as(state.h)/area
    rhs = replace(state, u_e=velocity+correction, h=h_rate, entropy_h=rates[..., 0]+source,
                  salt_h=rates[..., 1], tracer_h=None if state.tracer_h is None else rates[..., 2:])
    return rhs, WorkLedger(ledger, work, heat, source, correction, face['identity'][active])


def euler(model, state, dt, energy):
    rhs, work = evaluate(model, state, dt, energy)
    area, ledger = state.mesh.cell_area[:, None], work.transport
    amounts = ledger.amounts/area[..., None]
    end = replace(state, u_e=state.u_e+dt*rhs.u_e, h=ledger.volume/area,
        entropy_h=amounts[..., 0]+dt*work.entropy_source, salt_h=amounts[..., 1],
        tracer_h=None if state.tracer_h is None else amounts[..., 2:])
    # The full thermal domain, including heating, is checked here; a salt
    # positivity budget is not a theorem about nonlinear thermodynamic bounds.
    validate(model, end, energy)
    return end, work


def step(model, state, dt, energy):
    first, l1 = euler(model, state, dt, energy)
    second, l2 = euler(model, first, dt, energy)
    middle = .75*state+.25*second
    third, l3 = euler(model, middle, dt, energy)
    end = (1/3)*state+(2/3)*third
    validate(model, end, energy)
    return end, (l1, l2, l3)


def contraction(gradient, rhs):
    products = [gradient[k]*getattr(rhs, k) for k in ('u_e', 'h', 'entropy_h', 'salt_h')]
    return sum(x.sum() for x in products), sum(x.abs().sum() for x in products)
