"""Research temperature-coordinate dynamics; same H and spatial work pairing.

No inverse thermodynamics, altered EOS, or simplified caloric energy. Reuses
entropy transport/limiter algebra as a temporary computational coordinate, not
as prognostic storage. Finite RK steps are not invariant to this nonlinear
coordinate change. Optional TKE/forcing/ALE composition lives in system.py.
"""
import math
import torch
from vfs.ocean.stratified.equations import EntropyState, validate
from vfs.ocean.stratified import transport


def layer_fields(ocean, state):
    if state.mesh is not ocean.mesh or state.column is not ocean.column:
        raise ValueError('state must borrow model geometry')
    ref = ocean.reference.evaluate(state.salinity(), state.temperature(),
                                        entropy_derivatives=True)
    if (ref['entropy_theta'] <= 0).any():
        raise ValueError('temperature coordinate requires positive entropy derivative')
    entropy = EntropyState(state.u_e,state.h,state.h*ref['entropy'],state.salt_h,
                           state.tracer_h,state.mesh,state.column)
    validate(ocean.base,entropy,ocean.energy,physical=state)
    return entropy,ref


class LayerDynamics:
    """Advance thickness, velocity and conservative layer contents on matching levels."""
    def __init__(self, ocean):
        self.ocean = ocean

    def fields(self, state):
        return layer_fields(self.ocean, state)

    def energy(self, state):
        _,ref = self.fields(state)
        m = self.ocean.energy.mass_area(state)
        return self.ocean.energy.base.terms(state)['total']+(m*state.h*ref['h0']).sum()

    def gradients(self, state):
        _,ref = self.fields(state)
        g = self.ocean.energy.base.gradients(state)
        m = self.ocean.energy.mass_area(state)
        t,s = state.temperature(),state.salinity()
        return dict(u_e=g['u_e'],
            h=g['h']+m*(ref['h0']-t*ref['h_theta']-s*ref['h_s']),
            temp_h=g['temp_h']+m*ref['h_theta'],
            salt_h=g['salt_h']+m*ref['h_s'])

    def tendency_with_report(self, state, dt):
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        entropy,ref = self.fields(state)
        rhs,report = transport.evaluate(self.ocean.base,entropy,dt,self.ocean.energy,
                                         physical=state)
        t,s,eta = state.temperature(),state.salinity(),ref['entropy']
        # d(h eta) = eta_T d(h T) + eta_S d(h S)
        #            + (eta - T eta_T - S eta_S) dh.
        thermal = (rhs.entropy_h-ref['entropy_s']*rhs.salt_h
                   +(t*ref['entropy_theta']+s*ref['entropy_s']-eta)*rhs.h)/ref['entropy_theta']
        return state.with_tensors(rhs.u_e,rhs.h,thermal,rhs.salt_h,rhs.tracer_h),report

    def loading_acceleration(self, state, load, report):
        du = -state.mesh.grad(load) / state.column.rho0
        if self.ocean.domain is not None:
            du = self.ocean.domain.apply_edge(du)
        return du.expand_as(state.u_e)

    def euler(self, state, dt, *, extra_tendency=None):
        rhs,report = self.tendency_with_report(state,dt)
        area = state.mesh.cell_area[:,None]
        amounts = report.transport.amounts/area[...,None]
        # Retain authoritative bounded salt/passive/volume updates exactly.
        end = state.with_tensors(state.u_e+dt*rhs.u_e,report.transport.volume/area,
            state.temp_h+dt*rhs.temp_h,amounts[...,1],
            None if state.tracer_h is None else amounts[...,2:])
        # The authoritative endpoint is installed; validation does not need
        # the full flux/work report to remain live (autograd retains its graph).
        del rhs, report, amounts
        if extra_tendency is not None:
            # Both sources are evaluated on this same Euler input state.
            # Authoritative advective amounts remain the base update.
            du, dq = extra_tendency(state)
            end = end.with_tensors(end.u_e+dt*du, end.h, end.temp_h,
                                   end.salt_h, end.tracer_h+dt*dq)
            if not torch.isfinite(end.tracer_h).all() or (end.tracer_h[..., -2] < 0).any():
                raise FloatingPointError('combined transport/backscatter exhausted reservoir; reduce dt')
        self.fields(end)  # Reject out-of-domain stages; never clip temperature.
        return end

    def step(self, state, dt, *, extra_tendency=None):
        first = self.euler(state,dt,extra_tendency=extra_tendency)
        second = self.euler(first,dt,extra_tendency=extra_tendency)
        del first
        middle = .75*state+.25*second
        del second
        self.fields(middle)
        third = self.euler(middle,dt,extra_tendency=extra_tendency)
        del middle
        end = (1/3)*state+(2/3)*third
        del third
        self.fields(end)
        return end

# Historical name used by the temperature-coordinate comparison experiments.
TemperatureOcean = LayerDynamics
