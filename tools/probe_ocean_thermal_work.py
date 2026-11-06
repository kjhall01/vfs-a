"""Research caloric completion and first/second-law audit of bounded JMD95.

Explicit assumptions: constant reference heat capacity and idealized mixing
entropy. These are NOT supplied/calibrated by JMD95. No runtime/default change.
Signed local thermal exchange closes instantaneous total-energy work without
an all-mechanical correction. Entropy is audited, never silently repaired.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import vfs
from vfs.ocean.transport import divergence


def load_transport_probe():
    path = Path(__file__).with_name('probe_ocean_bounded_transport.py')
    spec = importlib.util.spec_from_file_location('bounded_transport_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transport_probe = load_transport_probe()


class CaloricCompletion:
    """Declared reduced-fluid thermodynamics, not empirical seawater calorics.

    H = H_JMD95 + rho0*cp*integral(h*theta); Celsius energy origin is arbitrary.
    eta(theta,S) = cp*log((theta+273.15)/283.15) - k*S*log(S/35).
    cp: J/(kg K); k: J/(kg K practical-salinity-unit). Both are explicit inputs.
    The S=0 entropy value is finite, but its salt derivative is singular.
    """
    def __init__(self, base, *, cp, salt_entropy_coefficient):
        if not math.isfinite(cp) or cp <= 0:
            raise ValueError('cp must be finite and positive')
        if not math.isfinite(salt_entropy_coefficient) or salt_entropy_coefficient < 0:
            raise ValueError('salt entropy coefficient must be finite and nonnegative')
        self.base, self.cp, self.k = base, float(cp), float(salt_entropy_coefficient)

    def mass_area(self, state):
        area = self.base.rho0*state.mesh.cell_area[:, None]
        return area if self.base.domain is None else area*self.base.domain.wet_cell[:, None]

    def check(self, state):
        if not torch.isfinite(state.temp_h).all() or (state.temperature()+273.15 <= 0).any():
            raise FloatingPointError('nonpositive/nonfinite absolute potential temperature')
        if not torch.isfinite(state.salt_h).all() or (state.salt_h < 0).any():
            raise FloatingPointError('salinity must be finite and nonnegative')

    def terms(self, state):
        self.check(state)
        result = self.base.terms(state)
        result['reference_thermal'] = (self.mass_area(state)*self.cp*state.temp_h).sum()
        result['total'] = result['total']+result['reference_thermal']
        return result

    def gradients(self, state):
        self.check(state)
        result = self.base.gradients(state)
        result['temp_h'] = result['temp_h']+self.mass_area(state)*self.cp
        return result

    def entropy(self, state):
        self.check(state)
        t, s = state.temperature()+273.15, state.salinity()
        eta = self.cp*torch.log(t/283.15)-self.k*torch.special.xlogy(s, s/35.)
        return (self.mass_area(state)*state.h*eta).sum()

    def entropy_gradients(self, state):
        self.check(state)
        t, s = state.temperature()+273.15, state.salinity()
        if (s <= 0).any():
            raise ValueError('instantaneous entropy audit requires positive salinity; use finite entropy at S=0')
        m = self.mass_area(state)
        return dict(u_e=torch.zeros_like(state.u_e),
            h=m*(self.cp*torch.log(t/283.15)-self.cp*(t-273.15)/t+self.k*s),
            temp_h=m*self.cp/t, salt_h=-m*self.k*(torch.log(s/35.)+1))

    def thermal_temperature(self, state):
        """dH/d(m*eta), Kelvin, for the declared reduced-fluid potential."""
        g = self.base.gradients(state)['temp_h']
        m = self.mass_area(state)
        safe = torch.where(m > 0, m, torch.ones_like(m))
        return (state.temperature()+273.15)*(1+g/(safe*self.cp))

    def thermodynamic_fields(self, state):
        """Reduced temperature and chemical potential of the declared fluid."""
        self.check(state)
        s = state.salinity()
        if (s <= 0).any():
            raise ValueError('chemical potential requires positive salinity')
        gradient = self.base.gradients(state)
        mass = self.mass_area(state)
        safe = torch.where(mass > 0, mass, torch.ones_like(mass))
        temperature = (state.temperature()+273.15)*(1+gradient['temp_h']/(safe*self.cp))
        chemical = gradient['salt_h']/safe+temperature*self.k*(torch.log(s/35.)+1)
        if not torch.isfinite(temperature).all() or (temperature <= 0).any():
            raise FloatingPointError('nonpositive reduced thermodynamic temperature')
        return temperature, chemical


@dataclass(frozen=True)
class WorkLedger:
    transport: object
    face_work: torch.Tensor
    thermal_power: torch.Tensor
    temperature_content_source: torch.Tensor


def thermodynamic_exchange(model, state, dt, caloric, *, heat_conductance, salt_mobility):
    """Separate fixed-geometry Onsager reference, NOT a transport replacement.

    On each active face: energy flux = K*(T_i-T_j), and salt flux =
    -M*(mu_j/T_j-mu_i/T_i). K is W/K; M is kg² PSU² K/(J s), per edge/layer.
    Positive K/M make both face entropy productions nonnegative. K/M are
    explicit research parameters, not calibrated diffusivities. Salt outflow
    is limited using the complete incident-face budget; positive rescaling
    preserves its entropy-production sign. S must be strictly positive.

    Transfers conserve total energy and reference-mass-weighted salt at the
    RHS level. Converting energy tendency to theta uses the full derivative,
    including the simultaneous salt change. A finite Euler update is not an
    exact nonlinear energy or entropy map. This does not repair FCT advection.
    """
    if caloric.base is not model.dynamics.energy:
        raise ValueError('caloric completion must borrow this model energy')
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if any(not math.isfinite(x) or x < 0 for x in (heat_conductance, salt_mobility)):
        raise ValueError('exchange coefficients must be finite and nonnegative')
    model.check_state(state)
    model.dynamics.check_state(state)
    temperature, chemical = caloric.thermodynamic_fields(state)
    _, i, j, _ = transport_probe.transport_for(model).graph(state)
    t, mu = temperature.flatten(), chemical.flatten()
    energy_flux = heat_conductance*(t[i]-t[j])
    chemical_jump = mu[j]/t[j]-mu[i]/t[i]
    salt_flux = -salt_mobility*chemical_jump
    rho_area = model.dynamics.rho0*model.mesh.cell_area[:, None]
    salt_amount = (rho_area*state.salt_h).flatten()
    out = torch.zeros_like(salt_amount).index_add(0, i, dt*salt_flux.clamp_min(0))
    out = out.index_add(0, j, dt*(-salt_flux).clamp_min(0))
    ratio = torch.where(out > 0, .9*salt_amount/torch.where(out > 0, out, torch.ones_like(out)),
                        torch.ones_like(out)).clamp(max=1)
    alpha = torch.where(salt_flux >= 0, ratio[i], ratio[j])
    salt_flux = alpha*salt_flux
    salt_rate = divergence(i, j, salt_flux, state.h.numel()).reshape_as(state.h)/rho_area
    energy_rate = divergence(i, j, energy_flux, state.h.numel()).reshape_as(state.h)
    gradient = caloric.gradients(state)
    wet = (caloric.mass_area(state) > 0).expand_as(state.h)
    coefficient = gradient['temp_h']
    if (coefficient[wet] <= 0).any():
        raise FloatingPointError('nonpositive thermal energy derivative')
    theta_rate = (energy_rate-gradient['salt_h']*salt_rate)/torch.where(wet, coefficient, torch.ones_like(coefficient))
    zero = state.zeros_like()
    rhs = zero.with_tensors(zero.u_e, zero.h, theta_rate, salt_rate, zero.tracer_h)
    # Stable quadratic products, avoiding subtraction of reciprocal temperatures.
    entropy_heat = heat_conductance*(t[i]-t[j]).square()/(t[i]*t[j])
    entropy_salt = alpha*salt_mobility*chemical_jump.square()
    return rhs, dict(energy_flux=energy_flux, salt_flux=salt_flux,
                    entropy_production=entropy_heat+entropy_salt, salt_limiter=alpha)


def evaluate(model, state, dt, caloric, scheme='fct'):
    """Close spatial work via signed thermal exchange, not a momentum force.

    Equal power to the two face endpoints is an explicit allocation choice.
    This first-law construction does NOT guarantee nonnegative entropy.
    """
    if caloric.base is not model.dynamics.energy:
        raise ValueError('caloric completion must borrow this model energy')
    centered = model.tendency(state)
    caloric.check(state)
    ledger = transport_probe.transport_for(model, scheme).advance(state, dt)
    rhs = transport_probe.ledger_rhs(state, ledger, dt, centered.u_e)
    base = caloric.base.gradients(state)
    area = state.mesh.cell_area[:, None]
    covector = torch.stack([base[k]/area for k in ('temp_h', 'salt_h')], -1).reshape(-1, 2)
    # cp contributes a constant thermal covector on wet nodes: its face jump
    # is exactly zero. Evaluate that cancellation analytically, not by
    # subtracting nearly equal cp + small_dynamic_enthalpy numbers.
    difference = (ledger.integrated_flux-ledger.high_integrated_flux)[:, :2]/dt
    work = ((covector[ledger.j]-covector[ledger.i])*difference).sum(-1)
    power = state.h.new_zeros(state.h.numel()).index_add(0, ledger.i, -.5*work)
    power = power.index_add(0, ledger.j, -.5*work).reshape_as(state.h)
    coefficient = base['temp_h']+caloric.mass_area(state)*caloric.cp
    wet = (caloric.mass_area(state) > 0).expand_as(state.h)
    if not torch.isfinite(coefficient).all() or (coefficient[wet] <= 0).any():
        raise FloatingPointError('nonpositive thermal energy derivative')
    source = power/torch.where(wet, coefficient, torch.ones_like(coefficient))
    result = rhs.with_tensors(rhs.u_e, rhs.h, rhs.temp_h+source, rhs.salt_h, rhs.tracer_h)
    return result, WorkLedger(ledger, work, power, source)


def euler(model, state, dt, caloric, scheme='fct'):
    rhs, ledger = evaluate(model, state, dt, caloric, scheme)
    transported = ledger.transport.install(state, velocity=state.u_e+dt*rhs.u_e)
    updated = transported.with_tensors(transported.u_e, transported.h,
        transported.temp_h+dt*ledger.temperature_content_source,
        transported.salt_h, transported.tracer_h)
    model.check_state(updated)
    model.dynamics.check_state(updated)
    caloric.check(updated)
    return updated, ledger


def step(model, state, dt, caloric, scheme='fct'):
    first, l1 = euler(model, state, dt, caloric, scheme)
    second, l2 = euler(model, first, dt, caloric, scheme)
    middle = .75*state+.25*second
    third, l3 = euler(model, middle, dt, caloric, scheme)
    updated = (1/3)*state+(2/3)*third
    model.check_state(updated)
    model.dynamics.check_state(updated)
    caloric.check(updated)
    return updated, (l1, l2, l3)


def contraction(gradients, rhs):
    local = [gradients[k]*getattr(rhs, k) for k in ('u_e', 'h', 'temp_h', 'salt_h')]
    return sum(x.sum() for x in local), sum(x.abs().sum() for x in local)


def stable_entropy_terms(caloric, state, ledger):
    """Face-form entropy audit, avoiding cancellation of global advective work.

    Algebraically independent of the nodal derivative/RHS contraction. Taylor
    forms evaluate log-mean differences near equal states; no entropy floor.
    """
    t = (state.temperature()+273.15).flatten()
    s = state.salinity().flatten()
    if (s <= 0).any():
        raise ValueError('face entropy audit requires positive salinity')
    flux = ledger.transport
    ti, tj, si, sj = t[flux.i], t[flux.j], s[flux.i], s[flux.j]
    xt, xs = (tj-ti)/(tj+ti), (sj-si)/(sj+si)
    # log(b/a) = 2*(x+x^3/3+x^5/5+...) with x=(b-a)/(b+a).
    def log_ratio(a, b, x):
        series = 2*x*(1+x.square()/3+x.pow(4)/5+x.pow(6)/7)
        return torch.where(x.abs() < 1e-3, series, torch.log(b)-torch.log(a))
    lt, ls = log_ratio(ti, tj, xt), log_ratio(si, sj, xs)
    central_t = torch.where(xt.abs() < 1e-3,
        -4*xt.pow(3)/3-8*xt.pow(5)/5-12*xt.pow(7)/7,
        lt-.5*(ti+tj)*(tj-ti)/(ti*tj))
    central_s = torch.where(xs.abs() < 1e-3,
        -(si+sj)*(xs.pow(3)/3+xs.pow(5)/5+xs.pow(7)/7),
        sj-si-.5*(si+sj)*ls)
    delta = flux.trace_change
    transport = caloric.base.rho0*flux.carrier*(
        caloric.cp*(central_t-delta[:, 0]*(tj-ti)/(ti*tj))
        +caloric.k*(central_s-delta[:, 1]*ls))
    thermal = caloric.entropy_gradients(state)['temp_h']*ledger.temperature_content_source
    return transport, thermal


def split_energy_change(caloric, before, after):
    """Report separate differences; never normalize by the large thermal pool."""
    dynamic = caloric.base.terms(after)['total']-caloric.base.terms(before)['total']
    thermal = (caloric.mass_area(before)*caloric.cp*(after.temp_h-before.temp_h)).sum()
    return dynamic, thermal


def make_case(kind='front', coast=False):
    model, state = transport_probe.active_case(coast, zero_salt=kind == 'zero_salt')
    x = model.mesh.cell_xyz[:, :1]
    if kind == 'salt_only':
        h = torch.tensor([500., 1000., 1500.])[None, :].expand_as(state.h).clone()
        state = model.state(u_e=state.u_e, h=h, temperature=torch.full_like(h, 10.),
                            salinity=35.+torch.where(x > 0, 1., -1.).expand_as(h), tracer=state.tracers())
    elif kind == 'tilted':
        h = torch.tensor([500., 1000., 1500.])[None, :]*(1+.01*x)
        state = model.state(u_e=state.u_e, h=h, temperature=(10.+1e-3*x).expand_as(h),
                            salinity=torch.full_like(h, 35.), tracer=state.tracers())
    if model.domain is not None:
        state = model.domain.project_state(state)
    return model, state


def audit_case(kind, coast, scheme):
    model, state = make_case(kind, coast)
    caloric = CaloricCompletion(model.dynamics.energy, cp=4000., salt_entropy_coefficient=.3)
    dt = 10.
    rhs, ledger = evaluate(model, state, dt, caloric, scheme)
    rate, scale = contraction(caloric.gradients(state), rhs)
    before_rate, _ = contraction(caloric.base.gradients(state), rhs)
    entropy_rate = None if kind == 'zero_salt' else float(contraction(caloric.entropy_gradients(state), rhs)[0])
    end, stages = step(model, state, dt, caloric, scheme)
    dynamic, thermal = split_energy_change(caloric, state, end)
    area = model.mesh.cell_area[:, None]
    wet_area = caloric.mass_area(state).sum()/model.dynamics.rho0
    weights = (1/6, 1/6, 2/3)
    source_inventory = dt*sum(w*(area*l.temperature_content_source).sum() for w, l in zip(weights, stages))
    actual_inventory = (area*(end.temp_h-state.temp_h)).sum()
    budgets0, budgets1 = model.budgets(state), model.budgets(end)
    return dict(kind=kind, coast=coast, scheme=scheme, dt_seconds=dt,
        energy_rate_W=float(rate), absolute_local_work_W=float(scale),
        energy_rate_normalized=float(rate/scale), dynamic_energy_rate_W=float(before_rate),
        limiter_work_W=float(ledger.face_work.sum()),
        thermal_supply_W=float(ledger.thermal_power.sum()),
        positive_face_work_W=float(ledger.face_work.clamp_min(0).sum()),
        negative_face_work_W=float(ledger.face_work.clamp_max(0).sum()),
        entropy_rate_W_K=entropy_rate,
        entropy_change_J_K=float(caloric.entropy(end)-caloric.entropy(state)),
        dynamic_energy_change_J=float(dynamic), reference_thermal_change_J=float(thermal),
        total_energy_change_J_m2=float((dynamic+thermal)/wet_area),
        temperature_content_change_m3_K=float(actual_inventory),
        predicted_temperature_content_change_m3_K=float(source_inventory),
        temperature_budget_error_m3_K=float(actual_inventory-source_inventory),
        maximum_temperature_source_K_s=float((ledger.temperature_content_source/state.h).abs().max()),
        relative_inventory_errors={k:float(((budgets1[k]-budgets0[k])/budgets0[k].abs()).abs().max())
                                   for k in ('volume', 'salinity_content', 'tracer_content')},
        minimum_stage_salinity=min(float((l.transport.amounts[..., 1]/l.transport.volume).min()) for l in stages))


def entropy_counterexamples():
    rows = []
    for kind in ('salt_only', 'tilted'):
        model, state = make_case(kind)
        for k in (0., .3):
            caloric = CaloricCompletion(model.dynamics.energy, cp=4000., salt_entropy_coefficient=k)
            for scheme in ('donor', 'fct'):
                rhs, ledger = evaluate(model, state, 10., caloric, scheme)
                bounded = rhs.with_tensors(rhs.u_e, rhs.h, rhs.temp_h-ledger.temperature_content_source,
                                          rhs.salt_h, rhs.tracer_h)
                eg = caloric.entropy_gradients(state)
                rate, scale = contraction(eg, rhs)
                transport_rate, _ = contraction(eg, bounded)
                heat_rate = (eg['temp_h']*ledger.temperature_content_source).sum()
                stable_transport, stable_thermal = stable_entropy_terms(caloric, state, ledger)
                rows.append(dict(kind=kind, salt_entropy_coefficient=k, scheme=scheme,
                    entropy_rate_W_K=float(rate), absolute_local_entropy_work_W_K=float(scale),
                    transport_entropy_rate_W_K=float(transport_rate), thermal_entropy_rate_W_K=float(heat_rate),
                    stable_entropy_rate_W_K=float(stable_transport.sum()+stable_thermal.sum()),
                    stable_transport_entropy_rate_W_K=float(stable_transport.sum()),
                    stable_local_entropy_work_W_K=float(stable_transport.abs().sum()+stable_thermal.abs().sum()),
                    limiter_work_W=float(ledger.face_work.sum())))
    return rows


def exchange_probe():
    rows = []
    for kind in ('salt_only', 'tilted'):
        for coast in (False, True):
            model, state = make_case(kind, coast)
            caloric = CaloricCompletion(model.dynamics.energy, cp=4000., salt_entropy_coefficient=.3)
            for multiplier in (1., 1e10):
                # Explicit conductances chosen to exercise ordinary and strongly
                # limited transfers, not to represent a measured ocean diffusivity.
                rhs, info = thermodynamic_exchange(model, state, 10., caloric,
                    heat_conductance=1e9, salt_mobility=multiplier*1e14)
                work, work_scale = contraction(caloric.gradients(state), rhs)
                entropy, entropy_scale = contraction(caloric.entropy_gradients(state), rhs)
                endpoint = state+10*rhs
                rows.append(dict(kind=kind, coast=coast, heat_conductance_W_K=1e9,
                    salt_mobility=multiplier*1e14, energy_rate_W=float(work),
                    absolute_local_work_W=float(work_scale), entropy_rate_W_K=float(entropy),
                    entropy_rate_from_faces_W_K=float(info['entropy_production'].sum()),
                    absolute_local_entropy_work_W_K=float(entropy_scale),
                    minimum_face_entropy_production_W_K=float(info['entropy_production'].min()),
                    minimum_salt_limiter=float(info['salt_limiter'].min()),
                    minimum_euler_salinity=float(endpoint.salinity().min()),
                    maximum_euler_salinity=float(endpoint.salinity().max()),
                    finite_euler_entropy_change_J_K=float(caloric.entropy(endpoint)-caloric.entropy(state)),
                    minimum_euler_absolute_potential_temperature_K=float((endpoint.temperature()+273.15).min())))
    return rows


def zero_velocity():
    model, initial = make_case('zero_salt')
    caloric = CaloricCompletion(model.dynamics.energy, cp=4000., salt_entropy_coefficient=.3)
    rows = []
    for scheme in ('donor', 'fct'):
        for eps in (1e-3, 1e-6, 1e-9, 1e-12, 0.):
            rhs_list, sources = [], []
            for sign in (-1, 1):
                state = initial.with_tensors(torch.full_like(initial.u_e, sign*eps), initial.h,
                                            initial.temp_h, initial.salt_h, initial.tracer_h)
                rhs, ledger = evaluate(model, state, 10., caloric, scheme)
                rhs_list.append(rhs)
                sources.append(ledger.temperature_content_source/state.h)
            rows.append(dict(scheme=scheme, epsilon_m_s=eps,
                force_jump_max_m_s2=float((rhs_list[1].u_e-rhs_list[0].u_e).abs().max()),
                thermal_source_max_K_s=max(float(x.abs().max()) for x in sources),
                thermal_source_jump_max_K_s=float((sources[1]-sources[0]).abs().max())))
    return rows


def timestep_sweep(quick=False):
    rows = []
    model, initial = make_case()
    # Amplified velocity resolves time-integration effects above the roundoff
    # associated with tiny thermal increments on a large background content.
    initial = initial.with_tensors(100*initial.u_e, initial.h, initial.temp_h,
                                   initial.salt_h, initial.tracer_h)
    caloric = CaloricCompletion(model.dynamics.energy, cp=4000., salt_entropy_coefficient=.3)
    for dt in ((160., 80.) if quick else (160., 80., 40., 20.)):
        state = initial
        for _ in range(round(640/dt)):
            state, _ = step(model, state, dt, caloric)
        dynamic, thermal = split_energy_change(caloric, initial, state)
        area = model.mesh.cell_area.sum()
        rows.append(dict(dt_seconds=dt, duration_seconds=640., velocity_multiplier=100., dynamic_change_J_m2=float(dynamic/area),
                         reference_thermal_change_J_m2=float(thermal/area),
                         total_change_J_m2=float((dynamic+thermal)/area)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    root = Path(vfs.__file__).parent
    with torch.no_grad():
        result = dict(scope='Research constant-cp/idealized-composition calorics; first law closed, second law audited, no accuracy gates',
            assumptions=dict(cp_J_kg_K=4000., salt_entropy_coefficient=.3, calibrated_seawater=False),
            torch_version=torch.__version__,
            runtime_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*.py'))},
            driver_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(transport_probe.__file__))},
            cases=[audit_case(kind, coast, scheme) for kind in ('front', 'zero_salt', 'salt_only', 'tilted')
                   for coast in (False, True) for scheme in ('donor', 'fct')],
            entropy_counterexamples=entropy_counterexamples(), zero_velocity=zero_velocity(),
            fixed_geometry_exchange=exchange_probe(),
            timestep=timestep_sweep(args.quick))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f'Thermal-work measurements written to {args.output}')


if __name__ == '__main__':
    main()
