"""Tensor-first interface. No interpolation, clipping or automatic device moves."""

import math
import torch
from .steppers import BoundedSSPRK3, CenteredIMEX
from .moist_conservative import MoistConservativeDynamics, MoistState
from .moist_cp import interface_pressure_mass
from .moist_budgets import inventories


class MoistAtmosphere(torch.nn.Module):
    """Own geometry/dynamics and delegate advancement to a supported stepper.

    column.phi_s is fixed surface geopotential (m²/s²), scalar or (cells,).
    Geometry and dynamics are registered once here; steppers only borrow them.
    Move this module before initializing states. Existing states remain ordinary
    tensor dataclasses; move them explicitly with state_to_model on shared geometry.
    """

    def __init__(self, mesh, column, *, stepper=None):
        super().__init__()
        self.mesh, self.column = mesh, column
        ref = mesh.cell_area
        for name in ("A_interface", "B_interface", "phi_s"):
            x = getattr(column, name)
            if x.dtype != ref.dtype or x.device != ref.device:
                raise ValueError("mesh and column must share dtype and device")
        if column.phi_s.shape not in (torch.Size([]), torch.Size([mesh.nCells])):
            raise ValueError("column.phi_s must be scalar or (cells,)")
        if not torch.isfinite(column.phi_s).all():
            raise ValueError("surface geopotential must be finite")
        self.dynamics = MoistConservativeDynamics()
        if stepper is None:
            stepper = BoundedSSPRK3()
        if type(stepper) not in (BoundedSSPRK3, CenteredIMEX):
            raise TypeError("stepper must be BoundedSSPRK3 or CenteredIMEX")
        stepper._bind(self.dynamics)
        self.stepper = stepper
        self.transport = stepper.transport if isinstance(stepper, BoundedSSPRK3) else None

    def __setattr__(self, name, value):
        # Fixed ownership: replacing a component could leave a collaborator stale.
        if name in ("mesh", "column", "dynamics", "stepper", "transport"):
            if name in self.__dict__ or name in self.__dict__.get("_modules", {}):
                raise AttributeError(f"{name} cannot be replaced; construct a new model")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in ("mesh", "column", "dynamics", "stepper", "transport"):
            raise AttributeError(f"{name} cannot be deleted; construct a new model")
        super().__delattr__(name)

    def _field(self, name, value, shape):
        """Broadcast supplied tensors/scalars while retaining their gradients."""
        ref = self.mesh.cell_area
        if not isinstance(value, (torch.Tensor, int, float)):
            raise TypeError(
                f"{name}: supply a torch.Tensor or scalar; convert external data explicitly"
            )
        if isinstance(value, torch.Tensor):
            if value.dtype != ref.dtype or value.device != ref.device:
                raise ValueError(f"{name}: dtype/device must match mesh")
            x = value
        else:
            x = ref.new_tensor(value)
        try:
            x = torch.broadcast_to(x, shape)
        except RuntimeError as error:
            raise ValueError(
                f"{name}: expected broadcastable to {shape}, got {tuple(x.shape)}"
            ) from error
        if not torch.isfinite(x).all():
            raise ValueError(f"{name}: values must be finite")
        return x

    def _surface_pressure(self, value):
        if isinstance(value, torch.Tensor) and value.ndim == 1:
            value = value[:, None]
        return self._field("surface_pressure", value, (self.mesh.nCells, 1))

    def interface_pressure(self, surface_pressure):
        """Pa, (cells, levels+1); ps may be scalar, (cells,) or (cells,1)."""
        p = self.column.interface_pressure(self._surface_pressure(surface_pressure))
        if (p <= 0).any() or (p.diff() <= 0).any():
            raise ValueError(
                "require positive interface pressures and layer thicknesses"
            )
        return p

    def normal_wind(self, *, eastward, northward):
        """Project winds already sampled at edges onto signed normals (m/s).

        Inputs broadcast to (edges, levels). This is not cell-to-edge interpolation.
        East/north are undefined at a geographic pole; supply normal wind there.
        """
        xyz = self.mesh.edge_xyz
        h = torch.linalg.vector_norm(xyz[:, :2], dim=-1)
        if (h <= torch.finfo(xyz.dtype).eps).any():
            raise ValueError("east/north basis undefined at a pole; supply normal wind")
        east = (
            torch.stack((-xyz[:, 1], xyz[:, 0], torch.zeros_like(h)), -1) / h[:, None]
        )
        north = torch.linalg.cross(xyz, east)
        shape = (xyz.shape[0], self.column.n_levels)
        u = self._field("eastward", eastward, shape)
        v = self._field("northward", northward, shape)
        return (
            u * (east * self.mesh.edge_normal).sum(-1)[:, None]
            + v * (north * self.mesh.edge_normal).sum(-1)[:, None]
        )

    def state_from_fields(
        self,
        *,
        temperature,
        surface_pressure,
        normal_wind,
        vapor,
        liquid=0.0,
        ice=0.0,
    ):
        """Create MoistState, preserving tensor autograd connections.

        T: K; ps: Pa; wind: m/s, (edges, levels).
        T and water fractions broadcast to (cells, levels+1), top to bottom.
        Fractions are kg species/kg TOTAL mixture, not dry-air mixing ratios.
        Scalars and vertical profiles broadcast; cell fields need (cells,1).
        Vapor must be strictly positive for the existing bounded moist kernel.
        Zero liquid/ice is supported. No phase equilibration, clipping, or
        regridding is performed.
        """
        p = self.interface_pressure(surface_pressure)
        ps = self._surface_pressure(surface_pressure)
        T = self._field("temperature", temperature, p.shape)
        water = torch.stack(
            [
                self._field(name, value, p.shape)
                for name, value in (("vapor", vapor), ("liquid", liquid), ("ice", ice))
            ],
            -1,
        )
        fractions = torch.cat((1 - water.sum(-1, keepdim=True), water), -1)
        self.dynamics.thermodynamics.validate(T, p, fractions)
        if (water[..., 0] <= 0).any():
            raise ValueError(
                "bounded moist kernel requires strictly positive vapor; no moisture floor is applied"
            )
        wind = self._field(
            "normal_wind",
            normal_wind,
            (self.mesh.edge_xyz.shape[0], self.column.n_levels),
        )
        mass = interface_pressure_mass(p.diff())
        return MoistState(
            wind,
            ps,
            mass * self.dynamics.thermodynamics.entropy(T, p, fractions),
            mass[..., None] * water,
            self.mesh,
            self.column,
        )

    def _check_geometry(self, state):
        if not isinstance(state, MoistState):
            raise TypeError("state must be a MoistState tensor dataclass")
        if state.mesh is not self.mesh or state.column is not self.column:
            raise ValueError("state must use this model's mesh and column")

    def _check_state(self, state):
        self._check_geometry(state)
        self.stepper._check_binding(self.dynamics)
        ref = self.mesh.cell_area
        for name in ("u_e", "ps", "entropy_content", "water_content"):
            value = getattr(state, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"state.{name} must be a tensor")
            if value.dtype != ref.dtype or value.device != ref.device:
                raise ValueError(
                    f"state.{name}: dtype/device must match model geometry "
                    f"({ref.dtype}, {ref.device}); use state_to_model explicitly"
                )

    def state_to_model(self, state):
        """Convert prognostic tensors to model dtype/device without detaching.

        Only states on these exact mesh/column objects are accepted; this is not
        regridding. Return a new dataclass, retaining tensors on a no-op conversion.
        Geometry is shared and was already moved by model.to(...).
        """
        self._check_geometry(state)
        self.stepper._check_binding(self.dynamics)
        ref = self.mesh.cell_area
        values = []
        for name in ("u_e", "ps", "entropy_content", "water_content"):
            value = getattr(state, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"state.{name} must be a tensor")
            values.append(value.to(device=ref.device, dtype=ref.dtype))
        return MoistState(*values, self.mesh, self.column)

    def physical_fields(self, state):
        """Recover interface T and fractions as tensors (not Gauss-point T)."""
        self._check_state(state)
        p = self.column.interface_pressure(state.ps)
        mass = interface_pressure_mass(p.diff())
        water = state.water_content / mass[..., None]
        fractions = torch.cat((1 - water.sum(-1, keepdim=True), water), -1)
        T = self.dynamics.thermodynamics.temperature_from_entropy(
            state.entropy_content / mass, p, fractions
        )
        return dict(
            temperature=T,
            surface_pressure=state.ps,
            normal_wind=state.u_e,
            vapor=water[..., 0],
            liquid=water[..., 1],
            ice=water[..., 2],
        )

    def forward(self, state, dt):
        """Advance by dt seconds using the configured existing algorithm.

        Bounded SSPRK3 retains each Euler species ledger. Centered IMEX has no
        bounded transport guarantee. Tendencies are separate numerical APIs.
        """
        self._check_state(state)
        if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive scalar in seconds")
        return self.stepper.advance(state, dt)

    def step(self, state, dt):
        """Equivalent to model(state, dt), including module forward hooks."""
        return self(state, dt)

    def diagnostics(self, state):
        """Global tensors: mass/species kg, entropy J/K, energy J incl. top reservoir."""
        self._check_state(state)
        return inventories(self.dynamics, state)
