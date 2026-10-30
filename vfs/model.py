"""Conservative dry atmosphere: fixed model ownership and explicit stepping."""
import math
import torch
from .core.state import State, PrognosticFields
from .atmosphere.conservative import DryHydrostaticConservativeDynamics
from .atmosphere.steppers import DrySSPRK3, DryIMEX


class DryAtmosphere(torch.nn.Module):
    """Own geometry and conservative dynamics once; states borrow geometry.

    State holds edge/full wind (m/s), surface pressure (Pa) and cell/interface
    pressure-weighted theta (Pa K). This core is unbatched and tracer-free.
    Topography is prescribed in column.phi_s; move the model before initializing.
    A stepper is a plain, single-model collaborator, never another module owner.
    """
    def __init__(self, mesh, column, *, dynamics=None, stepper=None):
        super().__init__()
        self.mesh, self.column = mesh, column
        ref = mesh.cell_area
        for name in ("A_interface", "B_interface", "phi_s"):
            value = getattr(column, name)
            if value.dtype != ref.dtype or value.device != ref.device:
                raise ValueError("mesh and column must share dtype/device")
        if column.phi_s.shape not in (torch.Size([]), torch.Size([mesh.nCells])):
            raise ValueError("column.phi_s must be scalar or (cells,)")
        if not torch.isfinite(column.phi_s).all():
            raise ValueError("surface geopotential must be finite")
        self.dynamics = DryHydrostaticConservativeDynamics() if dynamics is None else dynamics
        if type(self.dynamics) is not DryHydrostaticConservativeDynamics:
            raise TypeError("dynamics must be DryHydrostaticConservativeDynamics")
        self.stepper = DrySSPRK3() if stepper is None else stepper
        if type(self.stepper) not in (DrySSPRK3, DryIMEX):
            raise TypeError("stepper must be DrySSPRK3 or DryIMEX")
        self.stepper._bind(self.dynamics)

    def __setattr__(self, name, value):
        if name in ("mesh", "column", "dynamics", "stepper"):
            if name in self.__dict__ or name in self.__dict__.get("_modules", {}):
                raise AttributeError(f"{name} cannot be replaced; construct a new model")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in ("mesh", "column", "dynamics", "stepper"):
            raise AttributeError(f"{name} cannot be deleted; construct a new model")
        super().__delattr__(name)

    def bind_state(self, state):
        """Explicitly bind an unbound container; never regrid or mutate input."""
        if not isinstance(state, State):
            raise TypeError("state must be a dry State tensor container")
        for name in ("mesh", "column"):
            if getattr(state, name) is not None and getattr(state, name) is not getattr(self, name):
                raise ValueError(f"state {name} must be the model-owned object")
        if state.mesh is self.mesh and state.column is self.column:
            return state
        return State(fields=state.fields, mesh=self.mesh, column=self.column, attrs=state.attrs)

    def make_state(self, fields=None, *, u_e=None, mtheta_i=None, ps=None, tracer_m=None, attrs=None):
        fields = PrognosticFields(u_e, mtheta_i, ps, tracer_m) if fields is None else fields
        return State(fields=fields, mesh=self.mesh, column=self.column, attrs=attrs)

    def state_to_model(self, state):
        """Explicit differentiable dtype/device conversion on identical geometry."""
        if state.mesh is not self.mesh or state.column is not self.column:
            raise ValueError("state must borrow the model mesh and column")
        ref = self.mesh.cell_area
        return state.to(device=ref.device, dtype=ref.dtype)

    def _check_state(self, state):
        state = self.bind_state(state)
        self.stepper._check_binding(self.dynamics)
        ref = self.mesh.cell_area
        for name in ("u_e", "mtheta_i", "ps", "tracer_m"):
            value = getattr(state, name)
            if value is not None and (value.dtype != ref.dtype or value.device != ref.device):
                raise ValueError(f"state {name} dtype/device must match model; use state_to_model explicitly")
        self.dynamics.check_state_compatible(state)
        return state

    def forward(self, state, dt):
        state = self._check_state(state)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        return self.stepper.advance(state, dt)

    def step(self, state, dt):
        """One timestep through the module call, including forward hooks."""
        return self(state, dt)

    def tendency(self, state):
        """Instantaneous conservative RHS, distinct from time advancement."""
        return self.dynamics.tendency(self._check_state(state))

    def physical_fields(self, state):
        """Uncached dry fields: K/Pa on cell interfaces/full levels, wind m/s."""
        state = self._check_state(state)
        return dict(normal_wind=state.u_e, surface_pressure=state.ps,
                    potential_temperature=state.theta(), temperature=state.temperature(),
                    pressure_interface=state.interface_pressure(), pressure_full=state.full_pressure())

    def diagnostics(self, state):
        """Differentiable global budgets in kg and J, including fixed-lid energy."""
        state = self._check_state(state)
        return dict(mass=self.dynamics.total_mass(state), **self.dynamics.energy_terms(state),
                    boundary_total_energy=self.dynamics.boundary_total_energy(state))
