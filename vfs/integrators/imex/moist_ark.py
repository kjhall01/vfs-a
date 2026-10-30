"""Smooth moist ARK with analytic wave blocks and exact Schur elimination.

Centered transport only: stage admissibility is checked, never repaired. This
is not a bounded cloud transport scheme. No runtime AD or virtual-temperature
approximation. The native mixture-energy nonlinear RHS is unchanged.
"""
import math
import torch

from .solvers import (
    _gmres, IMEXSolverConfig, IMEXStageInfo, IMEXStepInfo, IMEXConvergenceError,
)
from .tableau import ARK4_TABLEAU
from .moist_wave import MoistPressureWave
from ...atmosphere.moist_conservative import MoistState
from ...atmosphere.moist_cp import reconstruct_moist_cp


class _Vector:
    """Private Krylov algebra; the public MoistState stays a tensor container."""
    def __init__(self, *values):
        self.values = values  # wind, surface pressure, entropy, water contents

    def __add__(self, other):
        return _Vector(*(a + b for a, b in zip(self.values, other.values)))

    def __sub__(self, other):
        return self + (-1) * other

    def __mul__(self, factor):
        return _Vector(*(factor * x for x in self.values))

    __rmul__ = __mul__

    def __truediv__(self, factor):
        return self * (1 / factor)

    def zeros_like(self):
        return _Vector(*(torch.zeros_like(x) for x in self.values))

    @classmethod
    def pack(cls, state):
        return cls(state.u_e, state.ps, state.entropy_content, state.water_content)

    def state(self, anchor):
        return MoistState(*self.values, anchor.mesh, anchor.column)


class _MoistInner:
    """Norm in wind, interface relative temperature, ps and water fractions.

    Species entropy-reference shifts change S but not these physical
    perturbations. Interface values also see the CP alternating mode that
    would be lost by a layer-average temperature norm.
    """
    def __init__(self, wave):
        self.w = wave
        self.tiny = torch.finfo(wave.s.u_e.dtype).tiny
        thermo = wave.d.thermodynamics
        q = torch.cat([
            1 - wave.scalars[..., 1:].sum(-1, keepdim=True),
            wave.scalars[..., 1:],
        ], -1)
        temperature = thermo.temperature_from_entropy(wave.scalars[..., 0], wave.p, q)
        gas_constant = thermo.gas_constant(q)
        self.cp = thermo.heat_capacity_p(q)
        pref = temperature.new_tensor([thermo.p_d_ref, thermo.p_v_ref, 1, 1])
        partial = wave.p[..., None] * q * wave.R / gas_constant[..., None]
        sa = (wave.cp * torch.log(temperature[..., None] / thermo.T_ref)
              + wave.href / thermo.T_ref
              - wave.R * torch.log(torch.where(partial > 0, partial, pref) / pref))
        self.eta = sa[..., 1:] - sa[..., :1]
        self.sd = sa[..., 0]
        self.pressure_factor = gas_constant / self.cp * wave.s.column.B_interface / wave.p

    def components(self, vector):
        u, p, S, C = vector.values
        w = self.w
        dm = w.dm1 * p
        thermal = ((S - (self.eta * C).sum(-1) - self.sd * dm) / (w.dm * self.cp)
                   + self.pressure_factor * p)
        q = (C - w.scalars[..., 1:] * dm[..., None]) / w.dm[..., None]
        return u / 10, p / 1e5, thermal, q / .01

    def dot(self, left, right):
        return sum((x * y).mean() for x, y in zip(self.components(left), self.components(right)))

    def norm(self, vector):
        return self.dot(vector, vector).clamp_min(0).sqrt()


class MoistPEIMEXARK:
    """Fixed-step smooth-mixture ARK; ARK4 is the default for this new class.

    The optional stage_validator(state, dt) is a read-only rejection hook,
    called at every actual stage and on the output. It receives the outer
    timestep, not a forward-Euler positivity budget. No limiter or physics
    provider is applied. Nonfinite or inadmissible states raise without
    changing the input. Supports eager analytic waves with full or Schur GMRES.
    """
    def __init__(self, dynamics, solver_config=None, *, stage_validator=None):
        self.dynamics = dynamics
        self.tableau = ARK4_TABLEAU
        self.solver_config = solver_config or IMEXSolverConfig(
            implicit_operator='gravity_wave', preconditioner='hydrostatic_schur',
            rtol=1e-9, max_iterations=80, restart=30,
        )
        self.solver_config.validate()
        cfg = self.solver_config
        if (cfg.implicit_operator != 'gravity_wave'
                or cfg.preconditioner not in ['none', 'hydrostatic_schur']
                or cfg.compile_kernels):
            raise ValueError('smooth moist ARK supports eager analytic waves, none/hydrostatic_schur only')
        self.last_step_info = None
        self.stage_validator = stage_validator

    def _validate(self, state, dt):
        if not all(bool(torch.isfinite(z).all()) for z in _Vector.pack(state).values):
            raise ValueError('nonfinite moist stage state')
        self.dynamics.transport_state(state)
        temperature = reconstruct_moist_cp(
            state.column.interface_pressure(state.ps), state.entropy_content,
            state.water_content, self.dynamics.thermodynamics,
        )['temperature']
        if not bool(torch.isfinite(temperature).all()) or bool((temperature <= 0).any()):
            raise ValueError('nonpositive/nonfinite moist stage temperature')
        if self.stage_validator is not None:
            self.stage_validator(state, dt)

    def step_with_info(self, state, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        self._validate(state, dt)
        wave = MoistPressureWave(self.dynamics, state)
        inner = _MoistInner(wave)
        anchor = _Vector.pack(state)
        zero = anchor.zeros_like()
        cfg = self.solver_config

        def transport(u):
            return _Vector(torch.zeros_like(state.u_e), *wave.transport(u))

        def pressure(z):
            return wave.wind(z.values[2], z.values[3], z.values[1])

        def linear(z):
            return _Vector(pressure(z), *transport(z.values[0]).values[1:])

        # Freeze L at the anchor, then split F(y) = [F(y)-L(y-y0)] + L(y-y0).
        explicit, implicit, infos = [], [], []
        for i in range(self.tableau.n_stages):
            known = zero
            for j in range(i):
                known = known + dt * (
                    self.tableau.explicit[i][j] * explicit[j]
                    + self.tableau.implicit[i][j] * implicit[j]
                )
            factor = dt * self.tableau.implicit[i][i]
            solve = None
            if factor == 0:
                correction = known
            else:
                rhs = known
                matvec = lambda z: z - factor * linear(z)
                if cfg.preconditioner == 'hydrostatic_schur':
                    # Eliminate u exactly: (I-a² T P) x = b_x + a T b_u.
                    rhs = _Vector(torch.zeros_like(state.u_e), *known.values[1:])
                    rhs = rhs + factor * transport(known.values[0])
                    matvec = lambda z: z - factor * factor * transport(pressure(z))
                solve = _gmres(
                    matvec, rhs, inner, rtol=cfg.resolved_rtol(state.u_e.dtype) * .1,
                    atol=cfg.atol * .1, max_iterations=cfg.max_iterations,
                    restart=cfg.restart, reorthogonalize=cfg.reorthogonalize,
                )
                correction = solve.solution
                if cfg.preconditioner == 'hydrostatic_schur':
                    correction = _Vector(
                        known.values[0] + factor * pressure(correction), *correction.values[1:],
                    )
            # The reduced tolerance alone is insufficient: check the full system.
            residual = inner.norm(correction - factor * linear(correction) - known)
            initial = inner.norm(known)
            target = cfg.atol + cfg.resolved_rtol(state.u_e.dtype) * initial
            converged = bool(residual <= target) and (solve is None or solve.converged)
            info = IMEXStageInfo(
                stage=i, implicit_coefficient=float(self.tableau.implicit[i][i]),
                iterations=0 if solve is None else solve.iterations,
                matvecs=0 if solve is None else solve.matvecs,
                preconditioner_applications=0, recycled_dimension=0,
                initial_residual=float(initial), final_residual=float(residual),
                relative_residual=float(residual / initial.clamp_min(inner.tiny)),
                converged=converged,
            )
            if not converged:
                raise IMEXConvergenceError(info)
            stage = (anchor + correction).state(state)
            if i > 0:  # The first stage is the already-validated anchor.
                self._validate(stage, dt)
            lin = linear(correction)
            explicit.append(_Vector.pack(self.dynamics.tendency(stage)) - lin)
            implicit.append(lin)
            infos.append(info)
        out = anchor
        for weight, ex, im in zip(self.tableau.weights, explicit, implicit):
            out = out + dt * weight * (ex + im)
        result = out.state(state)
        self._validate(result, dt)
        stepinfo = IMEXStepInfo(
            method=self.tableau.name, linear_operator='frozen_analytic_mixture_gravity_wave',
            preconditioner=cfg.preconditioner, dt=dt, stages=tuple(infos), converged=True,
            total_iterations=sum(x.iterations for x in infos),
            total_matvecs=sum(x.matvecs for x in infos), total_preconditioner_applications=0,
        )
        self.last_step_info = stepinfo
        return result, stepinfo

    def step(self, state, dt):
        return self.step_with_info(state, dt)[0]
