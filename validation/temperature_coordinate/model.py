"""Research temperature-coordinate dynamics; same H and spatial work pairing.

No inverse thermodynamics, altered EOS, or simplified caloric energy. Reuses
entropy transport/limiter algebra as a temporary computational coordinate, not
as prognostic storage. Finite RK steps are not invariant to this nonlinear
coordinate change. No TKE/forcing/ALE integration is claimed.
"""
import math
import torch
from vfs.ocean.stratified.equations import EntropyState, validate
from vfs.ocean.stratified import transport


class TemperatureOcean:
    """Wrap an existing StratifiedOcean; evolve OceanState.temp_h = h*theta."""
    def __init__(self, ocean):
        self.ocean = ocean

    def fields(self, state):
        if state.mesh is not self.ocean.mesh or state.column is not self.ocean.column:
            raise ValueError('state must borrow model geometry')
        ref = self.ocean.reference.evaluate(state.salinity(), state.temperature(),
                                            entropy_derivatives=True)
        if (ref['entropy_theta'] <= 0).any():
            raise ValueError('temperature coordinate requires positive entropy derivative')
        entropy = EntropyState(state.u_e,state.h,state.h*ref['entropy'],state.salt_h,
                               state.tracer_h,state.mesh,state.column)
        validate(self.ocean.base,entropy,self.ocean.energy,physical=state)
        return entropy,ref

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

    def euler(self, state, dt):
        rhs,report = self.tendency_with_report(state,dt)
        area = state.mesh.cell_area[:,None]
        amounts = report.transport.amounts/area[...,None]
        # Retain authoritative bounded salt/passive/volume updates exactly.
        end = state.with_tensors(state.u_e+dt*rhs.u_e,report.transport.volume/area,
            state.temp_h+dt*rhs.temp_h,amounts[...,1],
            None if state.tracer_h is None else amounts[...,2:])
        self.fields(end)  # Reject out-of-domain stages; never clip temperature.
        return end

    def step(self, state, dt):
        first = self.euler(state,dt)
        second = self.euler(first,dt)
        middle = .75*state+.25*second
        self.fields(middle)
        third = self.euler(middle,dt)
        end = (1/3)*state+(2/3)*third
        self.fields(end)
        return end
