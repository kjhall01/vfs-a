"""Pure tensor diagnostic term decompositions for dry dynamics."""

from __future__ import annotations

import torch

from ..core.state import State


# Numerical functions bound explicitly by DryHydrostaticDynamics.
def _zero_state_like(self, state: State) -> State:
    self.check_state_compatible(state)
    return State(
        torch.zeros_like(state.u_e),
        torch.zeros_like(state.mtheta_i),
        torch.zeros_like(state.ps),
        None if state.tracer_m is None else torch.zeros_like(state.tracer_m),
        mesh=state.mesh,
        column=state.column,
        attrs=dict(state.attrs or {}),
    )

def _state_with_tendency(self, state: State, tend: State, eps: float) -> State:
    self.check_state_compatible(state)
    tracer_m = None
    if state.tracer_m is not None and tend.tracer_m is not None:
        tracer_m = state.tracer_m + float(eps) * tend.tracer_m
    return State(
        state.u_e + float(eps) * tend.u_e,
        state.mtheta_i + float(eps) * tend.mtheta_i,
        state.ps + float(eps) * tend.ps,
        tracer_m,
        mesh=state.mesh,
        column=state.column,
        attrs=dict(state.attrs or {}),
    )

def _directional_energy_rate(self, state: State, tend: State, *, eps: float = 1.0) -> torch.Tensor:
    e0 = self.total_energy(state)
    e1 = self.total_energy(self._state_with_tendency(state, tend, float(eps)))
    return (e1 - e0) / float(eps)

def edge_metric(
    self,
    state: State,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Energy/force edge metric `dc_edge * dv_edge * cell_to_edge(dp)`."""
    self.check_state_compatible(state)
    if primitives is not None:
        dp_e = primitives.get("dp_e")
        if dp_e is None:
            dp_e = state.cell_to_edge(primitives["dp"])
    else:
        dp_e = state.cell_to_edge(state.layer_dp())
    return state.mesh.dc_edge[:, None] * state.mesh.dv_edge[:, None] * dp_e

def pressure_fields(self, state: State) -> dict[str, torch.Tensor]:
    """Return pressure fields used by the active column."""
    self.check_state_compatible(state)
    return {
        "ps": state.ps,
        "p_interface": state.interface_pressure(),
        "p_full": state.full_pressure(),
        "dp": state.layer_dp(),
    }

def thermo_fields(self, state: State) -> dict[str, torch.Tensor]:
    """Return thermodynamic fields from the active thermodynamics strategy."""
    self.check_state_compatible(state)
    return self.thermodynamics.thermo_fields(self, state)

def wind_fields(self, state: State) -> dict[str, torch.Tensor]:
    """Return wind fields and available reconstructions."""
    self.check_state_compatible(state)
    out = {"u_e": state.u_e}
    out["cell_wind_vector"] = state.cell_wind_vector()
    return out

def energy_terms(self, state: State) -> dict[str, torch.Tensor]:
    """Return integrated dry energy components."""
    kinetic = self.total_kinetic_energy(state)
    potential = self.total_potential_energy(state)
    internal = self.total_internal_energy(state)
    return {
        "kinetic": kinetic,
        "potential": potential,
        "internal": internal,
        "total": kinetic + potential + internal,
    }

def tracer_terms(self, state: State) -> dict[str, torch.Tensor]:
    """Return passive-tracer diagnostic tensors, or `{}` for dry states."""
    if state.tracer_m is None:
        return {}
    q = state.tracers()
    assert q is not None
    return {
        "q": q,
        "tracer_m": state.tracer_m,
        "total_tracer_mass": state.total_tracer_mass(),
    }

def static_stability(self, state: State) -> torch.Tensor:
    """
    Interface-theta static-stability margin.

    Vertical index increases downward, so positive values mean theta
    decreases downward and negative values indicate unstable reversals.
    """
    self.check_state_compatible(state)
    return state.static_stability()

def static_instability_penalty(self, state: State, margin: float = 0.0) -> torch.Tensor:
    """Differentiable mean-square penalty for static-stability violations."""
    stability = self.static_stability(state)
    return torch.relu(float(margin) - stability).square().mean()

def vertical_structure_terms(
    self,
    state: State,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compact vertical high-mode and baroclinic-structure metrics."""
    metric = self.edge_metric(state, primitives=primitives).clamp_min(torch.finfo(state.u_e.dtype).tiny)
    total_ke = (state.u_e.square() * metric).sum().clamp_min(torch.finfo(state.u_e.dtype).tiny)
    u_bar = (state.u_e * metric).sum(dim=-1, keepdim=True) / metric.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(metric.dtype).tiny)
    baroclinic_ke = ((state.u_e - u_bar).square() * metric).sum()
    if state.column.n_levels >= 3:
        u_second = state.u_e[..., :-2] - 2.0 * state.u_e[..., 1:-1] + state.u_e[..., 2:]
        u_second_rms = u_second.square().mean().sqrt()
    else:
        u_second_rms = torch.zeros((), dtype=state.u_e.dtype, device=state.u_e.device)
    theta = state.theta()
    if theta.shape[-1] >= 3:
        th_second = theta[..., :-2] - 2.0 * theta[..., 1:-1] + theta[..., 2:]
        th_second_rms = th_second.square().mean().sqrt()
    else:
        th_second_rms = torch.zeros((), dtype=theta.dtype, device=theta.device)
    return {
        "baroclinic_ke_fraction": baroclinic_ke / total_ke,
        "u_vertical_second_diff_rms": u_second_rms,
        "theta_vertical_second_diff_rms": th_second_rms,
    }

def mass_flux(self, state: State) -> torch.Tensor:
    """Full-level horizontal mass flux `u_e * cell_to_edge(dp)`."""
    self.check_state_compatible(state)
    return state.u_e * state.cell_to_edge(state.layer_dp())

def rotated_mass_flux(self, state: State) -> torch.Tensor:
    """Mass flux after the mesh-owned tangential reconstruction."""
    self.check_state_compatible(state)
    F = self.mass_flux(state)
    return state.tangential_velocity(F)

def rotation_term(self, state: State) -> torch.Tensor:
    """TRiSK vector-invariant rotation term in the u-equation."""
    self.check_state_compatible(state)
    return self._trisk_rotation(state)

def pgf_term(self, state: State) -> torch.Tensor:
    """Hydrostatic pressure-gradient term excluding the KE gradient."""
    self.check_state_compatible(state)
    return self._pgf_term_for_state(state, state.theta())

def bernoulli_term(self, state: State) -> torch.Tensor:
    """Kinetic-energy gradient part of the u-equation."""
    self.check_state_compatible(state)
    return -state.grad(state.ke_cell())

def mass_transport_term(self, state: State) -> torch.Tensor:
    """Surface-pressure transport tendency."""
    return self.ps_tendency_terms(state)["total"]

def mtheta_transport_term(self, state: State) -> torch.Tensor:
    """Potential-temperature mass transport excluding coordinate work."""
    terms = self.mtheta_tendency_terms(state)
    return terms["horizontal"] + terms["vertical"]

def tracer_transport_term(self, state: State) -> torch.Tensor | None:
    """Passive-tracer transport tendency, or None for dry states."""
    terms = self.tracer_tendency_terms(state)
    if terms is None:
        return None
    return terms["total"]

def coordinate_work_term(self, state: State) -> torch.Tensor:
    """Scalar coordinate-work correction in the mtheta equation."""
    return self.mtheta_tendency_terms(state)["coord_work"]

def tendency_terms(
    self,
    state: State,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return a nested tensor decomposition whose totals match `tendency`."""
    self.check_state_compatible(state)
    ctx = self._transport_primitives(state) if primitives is None else primitives

    out: dict[str, dict[str, torch.Tensor]] = {
        "u_e": self.momentum_tendency_terms(state, primitives=ctx),
        "mtheta_i": self.mtheta_tendency_terms(state, primitives=ctx),
        "ps": self.ps_tendency_terms(state, primitives=ctx),
    }

    tracer_terms = self.tracer_tendency_terms(state, primitives=ctx)
    if tracer_terms is not None:
        out["tracer_m"] = tracer_terms

    return out

def terms_total(self, terms: dict[str, dict[str, torch.Tensor]], *, like: State | None = None) -> State:
    """Collapse `tendency_terms` output to a `State`-shaped RHS.

    Pass ``like=state`` to preserve mesh, column, and attrs on the returned
    RHS. Omitting ``like`` keeps the historical contextless return value.
    """
    tracer_total = None
    if "tracer_m" in terms:
        tracer_total = terms["tracer_m"]["total"]
    mesh = None if like is None else like.mesh
    column = None if like is None else like.column
    attrs = {} if like is None else dict(like.attrs or {})
    return State(
        u_e=terms["u_e"]["total"],
        mtheta_i=terms["mtheta_i"]["total"],
        ps=terms["ps"]["total"],
        tracer_m=tracer_total,
        mesh=mesh,
        column=column,
        attrs=attrs,
    )

def force_terms(
    self,
    state: State,
    terms: dict[str, dict[str, torch.Tensor]] | None = None,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Return canonical momentum-force tensors and aliases used by reports."""
    if terms is None:
        terms = self.tendency_terms(state, primitives=primitives)
    u_terms = terms["u_e"]
    total = u_terms["total"]
    ctx = self._transport_primitives(state) if primitives is None else primitives
    metric = self.edge_metric(state, primitives=ctx)
    diffusion = u_terms.get("momentum_diffusion", u_terms.get("hyperdiffusion", torch.zeros_like(total)))
    out = {
        "rotation": u_terms["rotation"],
        "pgf": u_terms["pgf"],
        "hydro": u_terms["pgf"],
        "bernoulli": u_terms["bernoulli"],
        "minus_grad_ke": u_terms["bernoulli"],
        "vertical": u_terms["vertical"],
        "uvert": u_terms["vertical"],
        "hyperdiffusion": u_terms.get("hyperdiffusion", torch.zeros_like(total)),
        "divergence_damping": u_terms.get("divergence_damping", torch.zeros_like(total)),
        "momentum_diffusion": diffusion,
        "diffusion": diffusion,
        "total": total,
        "metric": metric,
        "dp": ctx["dp"],
        "dps_dt": ctx["dps_dt"],
        "layer_div": ctx["layer_div"],
        "mu": ctx["mu"],
        "omega": ctx["omega"],
    }
    if "top_sponge" in u_terms:
        out["top_sponge"] = u_terms["top_sponge"]
    return out

def force_balance_terms(
    self,
    state: State,
    terms: dict[str, dict[str, torch.Tensor]] | None = None,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Return weighted force-balance norms and cancellation ratio."""
    force = self.force_terms(state, terms=terms, primitives=primitives)
    metric = force["metric"].clamp_min(torch.finfo(state.u_e.dtype).tiny)

    def norm(name: str) -> torch.Tensor:
        return (force[name].square() * metric).sum().sqrt()

    rotation_norm = norm("rotation")
    pgf_norm = norm("pgf")
    bernoulli_norm = norm("bernoulli")
    vertical_norm = norm("vertical")
    total_norm = norm("total")
    denom = (rotation_norm + pgf_norm + bernoulli_norm + vertical_norm).clamp_min(torch.finfo(total_norm.dtype).tiny)
    return {
        "rotation_norm": rotation_norm,
        "pgf_norm": pgf_norm,
        "bernoulli_norm": bernoulli_norm,
        "vertical_norm": vertical_norm,
        "total_norm": total_norm,
        "cancellation_ratio": total_norm / denom,
    }

def rotation_work_terms(self, state: State) -> dict[str, torch.Tensor]:
    """Return pure tensor work diagnostics for the skew rotation operator."""
    self.check_state_compatible(state)
    rot = self._trisk_rotation(state)
    metric = state.mesh.dc_edge[:, None] * state.mesh.dv_edge[:, None]
    work = (state.u_e * rot * metric).sum()
    scale = (state.u_e * rot * metric).abs().sum().clamp_min(1.0e-300)
    return {
        "rotation": rot,
        "rotation_rms": rot.square().mean().sqrt(),
        "work": work,
        "work_abs_scale": scale,
        "work_relative": work.abs() / scale,
    }

def _edge_extensive_to_cell_layer(
    self,
    state: State,
    edge_value: torch.Tensor,
    *,
    mode: str = "side_area",
    cell_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Conservatively distribute edge/layer extensive values to adjacent cells."""

    mesh = state.mesh
    if edge_value.shape[-2] != mesh.nEdges:
        raise ValueError("edge_value must have shape [..., nEdges, nLayers]")
    n_layers = edge_value.shape[-1]
    leading = edge_value.shape[:-2]
    flat = edge_value.reshape(-1, mesh.nEdges, n_layers)
    batch = flat.shape[0]

    mode = mode.lower()
    if mode in ("side_area", "edge_side_area"):
        denom = mesh.edge_side_area.sum(dim=-1).clamp_min(1.0e-300).to(
            dtype=edge_value.dtype,
            device=edge_value.device,
        )
        w0 = (mesh.edge_side_area[:, 0].to(dtype=edge_value.dtype, device=edge_value.device) / denom).reshape(1, -1, 1)
        w1 = (mesh.edge_side_area[:, 1].to(dtype=edge_value.dtype, device=edge_value.device) / denom).reshape(1, -1, 1)
    elif mode == "equal":
        w0 = torch.full((1, mesh.nEdges, 1), 0.5, dtype=edge_value.dtype, device=edge_value.device)
        w1 = torch.full_like(w0, 0.5)
    elif mode == "cell_area":
        area = mesh.cell_area.to(dtype=edge_value.dtype, device=edge_value.device)
        a0 = area[mesh.cellsOnEdge[:, 0].to(device=edge_value.device)]
        a1 = area[mesh.cellsOnEdge[:, 1].to(device=edge_value.device)]
        denom = (a0 + a1).clamp_min(1.0e-300)
        w0 = (a0 / denom).reshape(1, -1, 1)
        w1 = (a1 / denom).reshape(1, -1, 1)
    elif mode in ("layer_mass", "mass"):
        if cell_weight is None:
            area_shape = [1] * edge_value.ndim
            area_shape[-2] = mesh.nCells
            area_shape[-1] = 1
            cell_weight = state.layer_dp() * mesh.cell_area.reshape(area_shape).to(
                dtype=edge_value.dtype,
                device=edge_value.device,
            )
        weight_flat = cell_weight.to(dtype=edge_value.dtype, device=edge_value.device).reshape(-1, mesh.nCells, n_layers)
        if weight_flat.shape[0] == 1 and batch != 1:
            weight_flat = weight_flat.expand(batch, -1, -1)
        if weight_flat.shape[0] != batch:
            raise ValueError("cell_weight leading dimensions must match edge_value leading dimensions")
        c0_index = mesh.cellsOnEdge[:, 0].to(device=edge_value.device)
        c1_index = mesh.cellsOnEdge[:, 1].to(device=edge_value.device)
        a0 = weight_flat[:, c0_index, :]
        a1 = weight_flat[:, c1_index, :]
        denom = (a0 + a1).clamp_min(1.0e-300)
        w0 = a0 / denom
        w1 = a1 / denom
    else:
        raise ValueError(
            "edge-to-cell attribution mode must be one of "
            "'side_area', 'equal', 'cell_area', or 'layer_mass'"
        )

    c0 = mesh.cellsOnEdge[:, 0].to(device=edge_value.device).reshape(1, -1, 1).expand(batch, -1, n_layers)
    c1 = mesh.cellsOnEdge[:, 1].to(device=edge_value.device).reshape(1, -1, 1).expand(batch, -1, n_layers)
    out = torch.zeros(
        (batch, mesh.nCells, n_layers),
        dtype=edge_value.dtype,
        device=edge_value.device,
    )
    out.scatter_add_(1, c0, flat * w0)
    out.scatter_add_(1, c1, flat * w1)
    return out.reshape(*leading, mesh.nCells, n_layers)

def _pressure_work_column_residual_stats(
    self,
    pgf_column: torch.Tensor,
    ps_column: torch.Tensor,
) -> dict[str, torch.Tensor]:
    residual = pgf_column + ps_column
    scale_column = torch.maximum(pgf_column.abs(), ps_column.abs()).clamp_min(1.0)
    global_pgf = pgf_column.sum()
    global_ps = ps_column.sum()
    global_residual = residual.sum()
    global_scale = torch.maximum(global_pgf.abs(), global_ps.abs()).clamp_min(1.0)
    return {
        "pgf_kinetic_sum": global_pgf,
        "ps_kinetic_mass_sum": global_ps,
        "residual_sum": global_residual,
        "residual_relative": global_residual.abs() / global_scale,
        "column_residual_rms": residual.square().mean().sqrt(),
        "column_relative_rms": (residual.abs() / scale_column).square().mean().sqrt(),
    }

def pressure_work_local_terms(
    self,
    state: State,
    terms: dict[str, dict[str, torch.Tensor]] | None = None,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Local pressure-work audit fields.

    Edge PGF work is first computed with the exact kinetic-energy gradient
    and then conservatively distributed to adjacent cells for map/profile
    inspection.  The ``ps`` kinetic-mass piece is the layer-local part of
    ``dE/dps`` associated only with changing the mass weighting of existing
    kinetic energy.
    """

    if terms is None:
        terms = self.tendency_terms(state, primitives=primitives)
    grad_u = self.kinetic_energy_gradient_u(state)
    pgf_edge_layer = grad_u * terms["u_e"]["pgf"]
    pgf_cell_layer = self._edge_extensive_to_cell_layer(state, pgf_edge_layer)

    ps_grad_local = self.scalar_pressure_energy_gradient_local_terms(state)
    ps_tend = terms["ps"]["total"]
    layer_grad = ps_grad_local["layer"]
    interface_grad = ps_grad_local["interface"]
    column_grad = ps_grad_local["column"]

    ps_layer = {
        name: grad * ps_tend
        for name, grad in layer_grad.items()
    }
    ps_interface = {
        name: grad * ps_tend
        for name, grad in interface_grad.items()
    }
    ps_column = {
        name: grad * ps_tend
        for name, grad in column_grad.items()
    }

    pgf_column = pgf_cell_layer.sum(dim=-1, keepdim=True)
    kinetic_mass_layer = ps_layer["kinetic_mass"]
    kinetic_mass_column = kinetic_mass_layer.sum(dim=-1, keepdim=True)
    pgf_plus_kinetic_mass_layer = pgf_cell_layer + kinetic_mass_layer
    pgf_plus_kinetic_mass_column = pgf_column + kinetic_mass_column
    pgf_plus_ps_total_column = pgf_column + ps_column["total"]
    scale_column = torch.maximum(pgf_column.abs(), kinetic_mass_column.abs()).clamp_min(1.0)

    return {
        "edge_layer": {
            "pgf_kinetic": pgf_edge_layer,
        },
        "cell_layer": {
            "pgf_kinetic": pgf_cell_layer,
            "ps_kinetic_mass": kinetic_mass_layer,
            "pgf_plus_ps_kinetic_mass": pgf_plus_kinetic_mass_layer,
        },
        "cell_interface": ps_interface,
        "cell_column": {
            "pgf_kinetic": pgf_column,
            "ps_kinetic_mass": kinetic_mass_column,
            "ps_potential": ps_column["potential"],
            "ps_internal": ps_column["internal"],
            "ps_theta_mass_coupling": ps_column["theta_mass_coupling"],
            "ps_total": ps_column["total"],
            "pgf_plus_ps_kinetic_mass": pgf_plus_kinetic_mass_column,
            "pgf_plus_ps_kinetic_mass_relative": pgf_plus_kinetic_mass_column.abs() / scale_column,
            "pgf_plus_ps_total": pgf_plus_ps_total_column,
        },
    }

def pressure_work_local_attribution_terms(
    self,
    state: State,
    terms: dict[str, dict[str, torch.Tensor]] | None = None,
    *,
    modes: tuple[str, ...] = ("side_area", "equal", "cell_area", "layer_mass"),
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """
    Compare conservative edge-to-cell attributions for local work maps.

    This is a bookkeeping diagnostic: global sums are unchanged by
    construction, but column-local residuals can depend on how edge work is
    assigned to neighboring cells. Large sensitivity means the map should
    not be over-interpreted as a pointwise physical exchange.
    """

    if terms is None:
        terms = self.tendency_terms(state, primitives=primitives)
    grad_u = self.kinetic_energy_gradient_u(state)
    pgf_edge_layer = grad_u * terms["u_e"]["pgf"]
    ps_grad_local = self.scalar_pressure_energy_gradient_local_terms(state)
    kinetic_mass_layer = ps_grad_local["layer"]["kinetic_mass"] * terms["ps"]["total"]
    kinetic_mass_column = kinetic_mass_layer.sum(dim=-1, keepdim=True)

    area_shape = [1] * state.layer_dp().ndim
    area_shape[-2] = state.mesh.nCells
    area_shape[-1] = 1
    layer_mass_weight = state.layer_dp() * state.mesh.cell_area.reshape(area_shape).to(
        dtype=state.ps.dtype,
        device=state.ps.device,
    )

    out: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for mode in modes:
        pgf_cell_layer = self._edge_extensive_to_cell_layer(
            state,
            pgf_edge_layer,
            mode=mode,
            cell_weight=layer_mass_weight if mode in ("layer_mass", "mass") else None,
        )
        pgf_column = pgf_cell_layer.sum(dim=-1, keepdim=True)
        residual_column = pgf_column + kinetic_mass_column
        out[mode] = {
            "cell_column": {
                "pgf_kinetic": pgf_column,
                "ps_kinetic_mass": kinetic_mass_column,
                "pgf_plus_ps_kinetic_mass": residual_column,
            },
            "global": self._pressure_work_column_residual_stats(pgf_column, kinetic_mass_column),
        }
    return out

def pressure_work_edge_exchange_terms(
    self,
    state: State,
    terms: dict[str, dict[str, torch.Tensor]] | None = None,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Edge-native PGF / surface-pressure kinetic-mass exchange audit.

    The ``ps`` kinetic-mass scalar rate is rewritten with the discrete
    grad/div adjoint identity instead of first forming a cell-local
    ``ps_tendency`` map.  This asks whether the PGF/divergence/energy-metric
    pairing is healthy before any edge-to-cell visualization choice.
    """

    ctx = self._transport_primitives(state) if primitives is None else primitives
    if terms is None:
        terms = self.tendency_terms(state, primitives=ctx)
    grad_u_terms = self.kinetic_energy_gradient_u_terms(state)
    pgf_edge_normal = grad_u_terms["normal"] * terms["u_e"]["pgf"]
    pgf_edge_tangential = grad_u_terms["tangential"] * terms["u_e"]["pgf"]
    pgf_edge_layer = pgf_edge_normal + pgf_edge_tangential

    ps_grad_local = self.scalar_pressure_energy_gradient_local_terms(state)
    ps_column_grad = ps_grad_local["column"]["kinetic_mass"]
    cell_shape = [1] * ps_column_grad.ndim
    cell_shape[-2] = state.mesh.nCells
    cell_shape[-1] = 1
    cell_area = state.mesh.cell_area.reshape(cell_shape).to(
        dtype=ps_column_grad.dtype,
        device=ps_column_grad.device,
    )
    ps_grad_density = ps_column_grad / cell_area.clamp_min(torch.finfo(ps_column_grad.dtype).tiny)
    grad_ps_density = state.grad(ps_grad_density)

    edge_shape = [1] * ctx["mass_flux"].ndim
    edge_shape[-2] = state.mesh.nEdges
    edge_shape[-1] = 1
    edge_metric = (
        state.mesh.dc_edge * state.mesh.dv_edge
    ).reshape(edge_shape).to(dtype=state.ps.dtype, device=state.ps.device)
    ps_edge_layer = edge_metric * ctx["mass_flux"] * grad_ps_density
    residual_edge_layer = pgf_edge_layer + ps_edge_layer
    residual_normal_edge_layer = pgf_edge_normal + ps_edge_layer

    scalar_ps_rate = (ps_column_grad * terms["ps"]["mass_flux"]).sum()
    edge_ps_rate = ps_edge_layer.sum()
    pgf_rate = pgf_edge_layer.sum()
    pgf_normal_rate = pgf_edge_normal.sum()
    pgf_tangential_rate = pgf_edge_tangential.sum()
    residual_rate = residual_edge_layer.sum()
    residual_normal_rate = residual_normal_edge_layer.sum()
    global_scale = torch.maximum(pgf_rate.abs(), edge_ps_rate.abs()).clamp_min(1.0)
    normal_global_scale = torch.maximum(pgf_normal_rate.abs(), edge_ps_rate.abs()).clamp_min(1.0)
    rms_scale = torch.maximum(
        pgf_edge_layer.square().mean().sqrt(),
        ps_edge_layer.square().mean().sqrt(),
    ).clamp_min(torch.finfo(state.ps.dtype).tiny)
    normal_rms_scale = torch.maximum(
        pgf_edge_normal.square().mean().sqrt(),
        ps_edge_layer.square().mean().sqrt(),
    ).clamp_min(torch.finfo(state.ps.dtype).tiny)
    abs_scale = torch.maximum(pgf_edge_layer.abs().sum(), ps_edge_layer.abs().sum()).clamp_min(1.0)
    normal_abs_scale = torch.maximum(pgf_edge_normal.abs().sum(), ps_edge_layer.abs().sum()).clamp_min(1.0)

    return {
        "edge_layer": {
            "pgf_kinetic": pgf_edge_layer,
            "pgf_kinetic_normal": pgf_edge_normal,
            "pgf_kinetic_tangential": pgf_edge_tangential,
            "ps_kinetic_mass_flux": ps_edge_layer,
            "pgf_plus_ps_kinetic_mass_flux": residual_edge_layer,
            "pgf_normal_plus_ps_kinetic_mass_flux": residual_normal_edge_layer,
        },
        "global": {
            "pgf_kinetic": pgf_rate,
            "pgf_kinetic_normal": pgf_normal_rate,
            "pgf_kinetic_tangential": pgf_tangential_rate,
            "ps_kinetic_mass_flux": edge_ps_rate,
            "residual": residual_rate,
            "residual_relative": residual_rate.abs() / global_scale,
            "normal_residual": residual_normal_rate,
            "normal_residual_relative": residual_normal_rate.abs() / normal_global_scale,
            "edge_residual_rms": residual_edge_layer.square().mean().sqrt(),
            "edge_residual_relative_rms": residual_edge_layer.square().mean().sqrt() / rms_scale,
            "edge_residual_abs_sum_relative": residual_edge_layer.abs().sum() / abs_scale,
            "edge_normal_residual_rms": residual_normal_edge_layer.square().mean().sqrt(),
            "edge_normal_residual_relative_rms": residual_normal_edge_layer.square().mean().sqrt() / normal_rms_scale,
            "edge_normal_residual_abs_sum_relative": residual_normal_edge_layer.abs().sum() / normal_abs_scale,
            "pgf_tangential_to_normal_rms": pgf_edge_tangential.square().mean().sqrt() / pgf_edge_normal.square().mean().sqrt().clamp_min(torch.finfo(state.ps.dtype).tiny),
            "scalar_ps_kinetic_mass": scalar_ps_rate,
            "edge_ps_kinetic_mass_flux": edge_ps_rate,
            "ps_flux_adjoint_mismatch": edge_ps_rate - scalar_ps_rate,
            "ps_flux_adjoint_mismatch_relative": (edge_ps_rate - scalar_ps_rate).abs() / scalar_ps_rate.abs().clamp_min(1.0),
        },
    }

def pressure_work_tangential_stencil_terms(
    self,
    state: State,
    terms: dict[str, dict[str, torch.Tensor]] | None = None,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Stencil-aware tangential KE-metric pressure-work diagnostic.

    The tangential part of the kinetic-energy gradient is nonlocal because
    ``ke_cell`` uses ``u_t = T u``.  Its global PGF work can be written as

        sum_e W_e (T u)_e (T PGF)_e

    or, equivalently, as the source-edge adjoint form

        sum_j [T^T(W T u)]_j PGF_j.

    This diagnostic exposes both edge attributions plus the full
    receiver-edge/stencil-edge contribution tensor.  It is for inspection
    only and does not change the dynamics.
    """

    ctx = self._transport_primitives(state) if primitives is None else primitives
    if terms is None:
        terms = self.tendency_terms(state, primitives=ctx)

    mesh = state.mesh
    pgf = terms["u_e"]["pgf"]
    weight = self._kinetic_energy_edge_weight(state)
    u_t = state.tangential_velocity(state.u_e)
    pgf_t = state.tangential_velocity(pgf)
    tangential_receiver = weight * u_t * pgf_t
    tangential_source = state.tangential_velocity_adjoint(weight * u_t) * pgf

    stencil = mesh.safe_tan_stencil.to(device=pgf.device)
    weights = mesh.tan_weights_masked.to(dtype=pgf.dtype, device=pgf.device)
    pgf_stencil = pgf[..., stencil, :]
    stencil_contribution = (weight * u_t)[..., :, None, :] * weights[..., None] * pgf_stencil
    tangential_receiver_from_stencil = stencil_contribution.sum(dim=-2)

    grad_u_terms = self.kinetic_energy_gradient_u_terms(state)
    normal_pgf = grad_u_terms["normal"] * pgf

    ps_grad_local = self.scalar_pressure_energy_gradient_local_terms(state)
    ps_column_grad = ps_grad_local["column"]["kinetic_mass"]
    cell_shape = [1] * ps_column_grad.ndim
    cell_shape[-2] = mesh.nCells
    cell_shape[-1] = 1
    cell_area = mesh.cell_area.reshape(cell_shape).to(dtype=ps_column_grad.dtype, device=ps_column_grad.device)
    ps_grad_density = ps_column_grad / cell_area.clamp_min(torch.finfo(ps_column_grad.dtype).tiny)
    grad_ps_density = state.grad(ps_grad_density)
    edge_shape = [1] * ctx["mass_flux"].ndim
    edge_shape[-2] = mesh.nEdges
    edge_shape[-1] = 1
    edge_metric = (mesh.dc_edge * mesh.dv_edge).reshape(edge_shape).to(dtype=state.ps.dtype, device=state.ps.device)
    ps_edge_layer = edge_metric * ctx["mass_flux"] * grad_ps_density

    source_residual = normal_pgf + tangential_source + ps_edge_layer
    receiver_residual = normal_pgf + tangential_receiver + ps_edge_layer
    normal_residual = normal_pgf + ps_edge_layer

    def rms(value: torch.Tensor) -> torch.Tensor:
        return value.square().mean().sqrt()

    def global_rel(total: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return total.abs() / torch.maximum(a.abs(), b.abs()).clamp_min(1.0)

    def rel_rms(residual: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return rms(residual) / torch.maximum(rms(a), rms(b)).clamp_min(torch.finfo(residual.dtype).tiny)

    source_sum = tangential_source.sum()
    receiver_sum = tangential_receiver.sum()
    stencil_sum = stencil_contribution.sum()
    source_residual_sum = source_residual.sum()
    receiver_residual_sum = receiver_residual.sum()
    normal_residual_sum = normal_residual.sum()
    normal_sum = normal_pgf.sum()
    ps_sum = ps_edge_layer.sum()
    source_receiver_diff = tangential_source - tangential_receiver
    receiver_stencil_mismatch = tangential_receiver - tangential_receiver_from_stencil
    source_scale = torch.maximum(rms(tangential_source), rms(tangential_receiver)).clamp_min(torch.finfo(state.ps.dtype).tiny)
    receiver_abs = tangential_receiver.abs().sum().clamp_min(1.0)
    source_abs = tangential_source.abs().sum().clamp_min(1.0)
    stencil_abs = stencil_contribution.abs().sum()

    return {
        "edge_layer": {
            "normal_pgf": normal_pgf,
            "ps_kinetic_mass_flux": ps_edge_layer,
            "tangential_source": tangential_source,
            "tangential_receiver": tangential_receiver,
            "tangential_receiver_from_stencil": tangential_receiver_from_stencil,
            "source_receiver_difference": source_receiver_diff,
            "normal_plus_ps": normal_residual,
            "source_residual": source_residual,
            "receiver_residual": receiver_residual,
        },
        "stencil": {
            "tangential_contribution": stencil_contribution,
        },
        "global": {
            "normal_pgf": normal_sum,
            "ps_kinetic_mass_flux": ps_sum,
            "tangential_source": source_sum,
            "tangential_receiver": receiver_sum,
            "tangential_stencil_sum": stencil_sum,
            "source_receiver_sum_mismatch": source_sum - receiver_sum,
            "source_receiver_sum_mismatch_relative": (source_sum - receiver_sum).abs() / source_sum.abs().clamp_min(1.0),
            "receiver_stencil_sum_mismatch": receiver_sum - stencil_sum,
            "receiver_stencil_sum_mismatch_relative": (receiver_sum - stencil_sum).abs() / receiver_sum.abs().clamp_min(1.0),
            "receiver_stencil_pointwise_mismatch_rms": rms(receiver_stencil_mismatch),
            "receiver_stencil_pointwise_mismatch_relative_rms": rms(receiver_stencil_mismatch) / rms(tangential_receiver).clamp_min(torch.finfo(state.ps.dtype).tiny),
            "source_receiver_difference_rms": rms(source_receiver_diff),
            "source_receiver_difference_relative_rms": rms(source_receiver_diff) / source_scale,
            "source_abs_sum": source_abs,
            "receiver_abs_sum": receiver_abs,
            "stencil_abs_sum": stencil_abs,
            "stencil_abs_to_receiver_abs": stencil_abs / receiver_abs,
            "stencil_abs_to_source_abs": stencil_abs / source_abs,
            "normal_residual": normal_residual_sum,
            "source_residual": source_residual_sum,
            "receiver_residual": receiver_residual_sum,
            "normal_residual_relative": global_rel(normal_residual_sum, normal_sum, ps_sum),
            "source_residual_relative": global_rel(source_residual_sum, normal_sum + source_sum, ps_sum),
            "receiver_residual_relative": global_rel(receiver_residual_sum, normal_sum + receiver_sum, ps_sum),
            "normal_residual_relative_rms": rel_rms(normal_residual, normal_pgf, ps_edge_layer),
            "source_residual_relative_rms": rel_rms(source_residual, normal_pgf + tangential_source, ps_edge_layer),
            "receiver_residual_relative_rms": rel_rms(receiver_residual, normal_pgf + tangential_receiver, ps_edge_layer),
        },
    }

def pressure_work_residual_structure_terms(
    self,
    state: State,
    terms: dict[str, dict[str, torch.Tensor]] | None = None,
    *,
    max_zonal_wavenumber: int = 6,
    max_spherical_degree: int = 3,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Spatial-structure diagnostics for local pressure-work residuals.

    The analyzed field is the column-integrated local residual
    ``PGF kinetic work + ps kinetic-mass work`` converted to an area
    density.  The diagnostics distinguish coherent low zonal wavenumbers
    from mesh-scale roughness; they do not change model equations.
    """

    local = self.pressure_work_local_terms(state, terms=terms, primitives=primitives)
    residual = local["cell_column"]["pgf_plus_ps_kinetic_mass"]
    if residual.shape[-2] != state.mesh.nCells:
        raise ValueError("local residual must have shape [..., nCells, 1]")

    dtype = residual.dtype
    device = residual.device
    tiny = torch.finfo(dtype).tiny
    cell_shape = [1] * residual.ndim
    cell_shape[-2] = state.mesh.nCells
    cell_shape[-1] = 1
    area = state.mesh.cell_area.reshape(cell_shape).to(dtype=dtype, device=device)
    area_sum = area.sum(dim=-2, keepdim=True).clamp_min(tiny)
    density = residual / area.clamp_min(tiny)
    mean = (density * area).sum(dim=-2, keepdim=True) / area_sum
    anomaly = density - mean
    anomaly_energy = (anomaly.square() * area).sum(dim=-2, keepdim=True)
    anomaly_rms = (anomaly_energy / area_sum).clamp_min(0.0).sqrt()

    smooth = state.edge_to_cell(state.cell_to_edge(anomaly))
    rough = anomaly - smooth
    rough_energy = (rough.square() * area).sum(dim=-2, keepdim=True)
    roughness_ratio = (rough_energy / anomaly_energy.clamp_min(tiny)).clamp_min(0.0).sqrt()

    grad_jump = state.grad(anomaly) * state.mesh.dc_edge.to(dtype=dtype, device=device)[:, None]
    edge_shape = [1] * grad_jump.ndim
    edge_shape[-2] = state.mesh.nEdges
    edge_shape[-1] = 1
    edge_area = state.mesh.edge_side_area.sum(dim=-1).reshape(edge_shape).to(dtype=dtype, device=device)
    edge_area_sum = edge_area.sum(dim=-2, keepdim=True).clamp_min(tiny)
    neighbor_jump_rms = ((grad_jump.square() * edge_area).sum(dim=-2, keepdim=True) / edge_area_sum).clamp_min(0.0).sqrt()
    neighbor_jump_ratio = neighbor_jump_rms / anomaly_rms.clamp_min(tiny)

    xyz = state.mesh.cell_xyz.to(dtype=dtype, device=device)
    lon = torch.atan2(xyz[:, 1], xyz[:, 0]).reshape(cell_shape)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    zonal: dict[str, torch.Tensor] = {}
    low_mode_energy = torch.zeros_like(anomaly_energy)
    for m in range(1, int(max_zonal_wavenumber) + 1):
        phase_arg = float(m) * lon
        cos_m = torch.cos(phase_arg)
        sin_m = torch.sin(phase_arg)
        denom_cos = (area * cos_m.square()).sum(dim=-2, keepdim=True).clamp_min(tiny)
        denom_sin = (area * sin_m.square()).sum(dim=-2, keepdim=True).clamp_min(tiny)
        a = (area * anomaly * cos_m).sum(dim=-2, keepdim=True) / denom_cos
        b = (area * anomaly * sin_m).sum(dim=-2, keepdim=True) / denom_sin
        component = a * cos_m + b * sin_m
        energy = (area * component.square()).sum(dim=-2, keepdim=True)
        low_mode_energy = low_mode_energy + energy
        prefix = f"m{m}"
        zonal[f"{prefix}_cos_coeff"] = a
        zonal[f"{prefix}_sin_coeff"] = b
        zonal[f"{prefix}_amplitude"] = (a.square() + b.square()).sqrt()
        zonal[f"{prefix}_phase_rad"] = torch.atan2(b, a)
        zonal[f"{prefix}_energy_fraction"] = energy / anomaly_energy.clamp_min(tiny)

    basis_terms: list[tuple[str, torch.Tensor]] = []
    if max_spherical_degree >= 1:
        basis_terms.extend([("l1_x", x), ("l1_y", y), ("l1_z", z)])
    if max_spherical_degree >= 2:
        basis_terms.extend(
            [
                ("l2_xy", x * y),
                ("l2_xz", x * z),
                ("l2_yz", y * z),
                ("l2_x2_minus_y2", x.square() - y.square()),
                ("l2_3z2_minus_1", 3.0 * z.square() - 1.0),
            ]
        )
    if max_spherical_degree >= 3:
        basis_terms.extend(
            [
                ("l3_z_5z2_minus_3", z * (5.0 * z.square() - 3.0)),
                ("l3_x_5z2_minus_1", x * (5.0 * z.square() - 1.0)),
                ("l3_y_5z2_minus_1", y * (5.0 * z.square() - 1.0)),
                ("l3_z_x2_minus_y2", z * (x.square() - y.square())),
                ("l3_xyz", x * y * z),
                ("l3_x_x2_minus_3y2", x * (x.square() - 3.0 * y.square())),
                ("l3_y_3x2_minus_y2", y * (3.0 * x.square() - y.square())),
            ]
        )
    spherical: dict[str, torch.Tensor] = {}
    if basis_terms:
        basis = torch.stack([term for _name, term in basis_terms], dim=-1)
        weight = state.mesh.cell_area.to(dtype=dtype, device=device)
        gram = basis.transpose(0, 1) @ (basis * weight[:, None])
        eye = torch.eye(gram.shape[0], dtype=dtype, device=device)
        gram = gram + torch.finfo(dtype).eps * gram.diagonal().abs().max().clamp_min(1.0) * eye
        flat_anomaly = anomaly.reshape(-1, state.mesh.nCells, 1)
        rhs = torch.einsum("ck,c,bcq->kbq", basis, weight, flat_anomaly)
        coeff = torch.linalg.solve(gram, rhs.reshape(gram.shape[0], -1)).reshape(gram.shape[0], flat_anomaly.shape[0], 1)
        fitted = torch.einsum("ck,kbq->bcq", basis, coeff).reshape_as(anomaly)
        spherical_energy = (fitted.square() * area).sum(dim=-2, keepdim=True)
        spherical["low_degree_energy_fraction"] = (spherical_energy / anomaly_energy.clamp_min(tiny)).squeeze(-1).squeeze(-1)
        for idx, (name, _term) in enumerate(basis_terms):
            spherical[f"{name}_coeff"] = coeff[idx].reshape(*anomaly.shape[:-2], 1, 1).squeeze(-1).squeeze(-1)
    else:
        spherical["low_degree_energy_fraction"] = torch.zeros_like(anomaly_energy).squeeze(-1).squeeze(-1)

    return {
        "fields": {
            "residual_density": density,
            "residual_density_anomaly": anomaly,
            "residual_density_smooth": smooth,
            "residual_density_rough": rough,
        },
        "global": {
            "residual_sum": residual.sum(),
            "density_mean": mean.squeeze(-1).squeeze(-1),
            "density_anomaly_rms": anomaly_rms.squeeze(-1).squeeze(-1),
            "roughness_ratio": roughness_ratio.squeeze(-1).squeeze(-1),
            "neighbor_jump_rms": neighbor_jump_rms.squeeze(-1).squeeze(-1),
            "neighbor_jump_ratio": neighbor_jump_ratio.squeeze(-1).squeeze(-1),
            "low_zonal_wavenumber_energy_fraction": (low_mode_energy / anomaly_energy.clamp_min(tiny)).squeeze(-1).squeeze(-1),
        },
        "zonal_wavenumber": zonal,
        "spherical_harmonic": spherical,
    }

def energy_exchange_terms(
    self,
    state: State,
    terms: dict[str, dict[str, torch.Tensor]] | None = None,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Closed-form energy-rate table for named dry tendency packages.

    This diagnostic intentionally keeps the model equations untouched.  It
    asks where the dry-energy directional derivative lives when the RHS is
    decomposed into named wind, mtheta, and ps terms.  The ``wind_kinetic``
    table uses the exact gradient of ``total_kinetic_energy`` with respect
    to edge wind; ``wind_simple`` keeps the historical diagonal edge metric
    for comparison.
    """

    if terms is None:
        terms = self.tendency_terms(state, primitives=primitives)

    metric = self.edge_metric(state, primitives=primitives)
    grad_u = self.kinetic_energy_gradient_u(state)
    wind_simple = {
        name: (state.u_e * value * metric).sum() / float(self.gravity)
        for name, value in terms["u_e"].items()
    }
    wind_simple_abs_scale = {
        name: (state.u_e * value * metric / float(self.gravity)).abs().sum().clamp_min(1.0e-300)
        for name, value in terms["u_e"].items()
    }
    wind_simple_relative = {
        name: wind_simple[name].abs() / wind_simple_abs_scale[name]
        for name in wind_simple
    }
    wind_kinetic = {
        name: (grad_u * value).sum()
        for name, value in terms["u_e"].items()
    }
    wind_kinetic_abs_scale = {
        name: (grad_u * value).abs().sum().clamp_min(1.0e-300)
        for name, value in terms["u_e"].items()
    }
    wind_kinetic_relative = {
        name: wind_kinetic[name].abs() / wind_kinetic_abs_scale[name]
        for name in wind_kinetic
    }
    scalar_rates = self.scalar_energy_rate_terms_closed(state, terms["mtheta_i"], terms["ps"])
    ps_decomposed = self.scalar_pressure_energy_rate_terms_closed(state, terms["ps"])
    mtheta_rates = scalar_rates["mtheta_i"]
    ps_rates = scalar_rates["ps"]
    assert isinstance(mtheta_rates, dict)
    assert isinstance(ps_rates, dict)
    ps_mass_flux_by_piece = {
        name: rates["mass_flux"]
        for name, rates in ps_decomposed.items()
    }

    zero = torch.zeros((), dtype=state.ps.dtype, device=state.ps.device)

    def pair(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        total = zero
        for value in values:
            total = total + value
        return total

    def rel(total: torch.Tensor, values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        scale = torch.ones_like(total)
        for value in values:
            scale = torch.maximum(scale, value.abs())
        return total.abs() / scale

    pairing_specs = {
        "pgf_kinetic_plus_ps_mass_flux": (
            wind_kinetic["pgf"],
            ps_rates["mass_flux"],
        ),
        "pgf_kinetic_plus_ps_kinetic_mass": (
            wind_kinetic["pgf"],
            ps_mass_flux_by_piece["kinetic_mass"],
        ),
        "pgf_kinetic_plus_ps_potential": (
            wind_kinetic["pgf"],
            ps_mass_flux_by_piece["potential"],
        ),
        "pgf_kinetic_plus_ps_internal": (
            wind_kinetic["pgf"],
            ps_mass_flux_by_piece["internal"],
        ),
        "pgf_kinetic_plus_ps_theta_mass_coupling": (
            wind_kinetic["pgf"],
            ps_mass_flux_by_piece["theta_mass_coupling"],
        ),
        "pgf_kinetic_plus_ps_nonkinetic": (
            wind_kinetic["pgf"],
            ps_mass_flux_by_piece["potential"],
            ps_mass_flux_by_piece["internal"],
            ps_mass_flux_by_piece["theta_mass_coupling"],
        ),
        "pgf_kinetic_plus_scalar_total": (
            wind_kinetic["pgf"],
            scalar_rates["total"],
        ),
        "pgf_kinetic_plus_ps_mass_flux_plus_mtheta_coord_work": (
            wind_kinetic["pgf"],
            ps_rates["mass_flux"],
            mtheta_rates["coord_work"],
        ),
        "pgf_simple_plus_scalar_total": (
            wind_simple["pgf"],
            scalar_rates["total"],
        ),
        "bernoulli_kinetic_plus_mtheta_horizontal": (
            wind_kinetic["bernoulli"],
            mtheta_rates["horizontal"],
        ),
        "vertical_kinetic_plus_mtheta_vertical": (
            wind_kinetic["vertical"],
            mtheta_rates["vertical"],
        ),
        "vertical_kinetic_plus_mtheta_vertical_plus_coord_work": (
            wind_kinetic["vertical"],
            mtheta_rates["vertical"],
            mtheta_rates["coord_work"],
        ),
        "rotation_kinetic": (wind_kinetic["rotation"],),
        "rotation_simple": (wind_simple["rotation"],),
        "full_closed": (
            wind_kinetic["total"],
            scalar_rates["total"],
        ),
        "full_simple_wind_metric": (
            wind_simple["total"],
            scalar_rates["total"],
        ),
    }
    pairings = {name: pair(values) for name, values in pairing_specs.items()}
    pairing_relative = {
        name: rel(pairings[name], values)
        for name, values in pairing_specs.items()
    }

    return {
        "wind_simple": wind_simple,
        "wind_simple_abs_scale": wind_simple_abs_scale,
        "wind_simple_relative": wind_simple_relative,
        "wind_kinetic": wind_kinetic,
        "wind_kinetic_abs_scale": wind_kinetic_abs_scale,
        "wind_kinetic_relative": wind_kinetic_relative,
        "mtheta": mtheta_rates,
        "ps": ps_rates,
        "ps_decomposed": ps_decomposed,
        "scalar": {
            "mtheta_total": scalar_rates["mtheta_total"],
            "ps_total": scalar_rates["ps_total"],
            "total": scalar_rates["total"],
        },
        "pairings": pairings,
        "pairing_relative": pairing_relative,
    }

def pressure_work_terms(
    self,
    state: State,
    terms: dict[str, dict[str, torch.Tensor]] | None = None,
    *,
    eps: float = 1.0,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Return tensor pressure-work / scalar-coordinate energy exchange terms."""
    if terms is None:
        terms = self.tendency_terms(state, primitives=primitives)
    metric = self.edge_metric(state, primitives=primitives)
    pgf_work = (state.u_e * terms["u_e"]["pgf"] * metric).sum() / float(self.gravity)
    wind_rates_closed = self.wind_energy_rate_terms_closed(state, terms["u_e"])
    pgf_work_kinetic = wind_rates_closed["pgf"]
    scalar_tend = State(
        torch.zeros_like(state.u_e),
        terms["mtheta_i"]["total"],
        terms["ps"]["total"],
        None,
    )
    scalar_work = self._directional_energy_rate(state, scalar_tend, eps=eps)
    scalar_rates_closed = self.scalar_energy_rate_terms_closed(state, terms["mtheta_i"], terms["ps"])
    scalar_work_closed = scalar_rates_closed["total"]
    coord_tend = State(
        torch.zeros_like(state.u_e),
        terms["mtheta_i"]["coord_work"],
        torch.zeros_like(state.ps),
        None,
    )
    coord_work = self._directional_energy_rate(state, coord_tend, eps=eps)
    full_rate = self._directional_energy_rate(state, self.terms_total(terms, like=state), eps=eps)
    full_rate_closed = wind_rates_closed["total"] + scalar_work_closed
    residual = pgf_work + scalar_work
    residual_kinetic = pgf_work_kinetic + scalar_work
    residual_closed = pgf_work + scalar_work_closed
    residual_kinetic_closed = pgf_work_kinetic + scalar_work_closed
    residual_abs = residual.abs()
    residual_kinetic_abs = residual_kinetic.abs()
    residual_closed_abs = residual_closed.abs()
    residual_kinetic_closed_abs = residual_kinetic_closed.abs()
    pair_scale = torch.maximum(
        torch.maximum(pgf_work.abs(), scalar_work.abs()),
        torch.ones_like(residual_abs),
    )
    kinetic_pair_scale = torch.maximum(
        torch.maximum(pgf_work_kinetic.abs(), scalar_work.abs()),
        torch.ones_like(residual_kinetic_abs),
    )
    closed_pair_scale = torch.maximum(
        torch.maximum(pgf_work.abs(), scalar_work_closed.abs()),
        torch.ones_like(residual_closed_abs),
    )
    kinetic_closed_pair_scale = torch.maximum(
        torch.maximum(pgf_work_kinetic.abs(), scalar_work_closed.abs()),
        torch.ones_like(residual_kinetic_closed_abs),
    )
    out = {
        "pgf_work": pgf_work,
        "pgf_work_simple": pgf_work,
        "pgf_work_kinetic": pgf_work_kinetic,
        "scalar_work": scalar_work,
        "scalar_work_closed": scalar_work_closed,
        "coord_work": coord_work,
        "residual": residual,
        "residual_abs": residual_abs,
        "residual_rel_current": residual_abs / pair_scale,
        "residual_kinetic": residual_kinetic,
        "residual_kinetic_abs": residual_kinetic_abs,
        "residual_kinetic_rel_current": residual_kinetic_abs / kinetic_pair_scale,
        "residual_closed": residual_closed,
        "residual_closed_abs": residual_closed_abs,
        "residual_closed_rel_current": residual_closed_abs / closed_pair_scale,
        "residual_kinetic_closed": residual_kinetic_closed,
        "residual_kinetic_closed_abs": residual_kinetic_closed_abs,
        "residual_kinetic_closed_rel_current": residual_kinetic_closed_abs / kinetic_closed_pair_scale,
        "total_energy_directional_derivative": full_rate,
        "total_energy_rate_closed": full_rate_closed,
    }
    for name, rate in wind_rates_closed.items():
        out[f"wind_work_kinetic_{name}"] = rate
    mtheta_rates = scalar_rates_closed["mtheta_i"]
    ps_rates = scalar_rates_closed["ps"]
    assert isinstance(mtheta_rates, dict)
    assert isinstance(ps_rates, dict)
    for name, rate in mtheta_rates.items():
        out[f"scalar_work_closed_mtheta_{name}"] = rate
    for name, rate in ps_rates.items():
        out[f"scalar_work_closed_ps_{name}"] = rate
    out["scalar_work_closed_mtheta_total"] = scalar_rates_closed["mtheta_total"]
    out["scalar_work_closed_ps_total"] = scalar_rates_closed["ps_total"]
    return out

def cfl_terms(
    self,
    state: State,
    dt: float,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Return CFL fields and maxima for the current state."""
    ctx = self._transport_primitives(state) if primitives is None else primitives
    dp = ctx["dp"]
    dm = ctx["dm"]
    omega = ctx["omega"]
    mu = ctx["mu"]
    vertical = float(dt) * omega.abs() / dp.clamp_min(1.0)
    vertical_interface = float(dt) * mu.abs() / dm.clamp_min(torch.finfo(dm.dtype).tiny)
    horizontal = float(dt) * state.u_e.abs() / state.mesh.dc_edge[:, None].clamp_min(1.0)
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "vertical_interface": vertical_interface,
        "horizontal_max": horizontal.max(),
        "vertical_max": vertical.max(),
        "vertical_interface_max": vertical_interface.max(),
    }

def tracer_no_feedback_terms(self, dry_state: State, tracer_state: State) -> dict[str, torch.Tensor]:
    """Return dry-tendency differences between dry and tracer-carrying states."""
    dry = self.tendency(dry_state)
    traced = self.tendency(tracer_state)
    du = (traced.u_e - dry.u_e).abs().max()
    dmtheta = (traced.mtheta_i - dry.mtheta_i).abs().max()
    dps = (traced.ps - dry.ps).abs().max()
    max_diff = torch.maximum(torch.maximum(du, dmtheta), dps)
    return {
        "u_max_abs_diff": du,
        "mtheta_max_abs_diff": dmtheta,
        "ps_max_abs_diff": dps,
        "max_abs_diff": max_diff,
    }

def divergence(
    self,
    state: State,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Mass-flux divergence per full level."""
    self.check_state_compatible(state)
    if primitives is not None:
        return primitives["layer_div"]
    return self._layer_divergence(state, state.u_e, state.layer_dp())

def vorticity(self, state: State) -> torch.Tensor:
    """Vertex relative vorticity per full level."""
    self.check_state_compatible(state)
    return state.curl(state.u_e)

def omega(
    self,
    state: State,
    terms: dict[str, dict[str, torch.Tensor]] | None = None,
    *,
    primitives: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Full-level vertical mass flux.

    The optional ``terms`` argument is accepted for compatibility but does
    not carry hidden metadata; omega is recomputed explicitly from state.
    """
    ctx = self._transport_primitives(state) if primitives is None else primitives
    return ctx["omega"]

def tendency_scale_terms(
    self,
    state: State,
    terms: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, torch.Tensor]:
    """Return RHS norms and simple state/RHS timescale proxies."""
    if terms is None:
        terms = self.tendency_terms(state)

    def rms(x: torch.Tensor) -> torch.Tensor:
        return x.square().mean().sqrt()

    out = {
        "u_e_rhs_rms": rms(terms["u_e"]["total"]),
        "ps_rhs_rms": rms(terms["ps"]["total"]),
        "mtheta_i_rhs_rms": rms(terms["mtheta_i"]["total"]),
        "u_e_timescale": state.u_e.square().mean().sqrt() / rms(terms["u_e"]["total"]).clamp_min(torch.finfo(state.u_e.dtype).tiny),
        "ps_timescale": state.ps.abs().mean() / rms(terms["ps"]["total"]).clamp_min(torch.finfo(state.ps.dtype).tiny),
        "mtheta_i_timescale": state.mtheta_i.abs().mean() / rms(terms["mtheta_i"]["total"]).clamp_min(torch.finfo(state.mtheta_i.dtype).tiny),
    }
    if state.tracer_m is not None and "tracer_m" in terms:
        out["tracer_m_rhs_rms"] = rms(terms["tracer_m"]["total"])
        out["tracer_m_timescale"] = state.tracer_m.abs().mean() / rms(terms["tracer_m"]["total"]).clamp_min(torch.finfo(state.tracer_m.dtype).tiny)
    return out

def adjustment_shock_terms(
    self,
    state: State,
    terms: dict[str, dict[str, torch.Tensor]] | None = None,
    dt: float | None = None,
    primitives: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Generic model-adjustment shock primitives."""
    if terms is None:
        terms = self.tendency_terms(state, primitives=primitives)
    ctx = self._transport_primitives(state) if primitives is None else primitives
    balance = self.force_balance_terms(state, terms=terms, primitives=ctx)
    div = ctx["layer_div"]
    omega = ctx["omega"]
    vertical = self.vertical_structure_terms(state, primitives=ctx)
    score = (
        balance["cancellation_ratio"]
        + div.square().mean().sqrt()
        + omega.square().mean().sqrt()
        + vertical["u_vertical_second_diff_rms"]
    )
    out = {
        "force_cancellation_ratio": balance["cancellation_ratio"],
        "divergence_rms": div.square().mean().sqrt(),
        "dpsdt_rms": terms["ps"]["total"].square().mean().sqrt(),
        "omega_rms": omega.square().mean().sqrt(),
        "u_vertical_second_diff_rms": vertical["u_vertical_second_diff_rms"],
        "shock_score": score,
    }
    if dt is not None:
        out["ps_step_fraction_max"] = (float(dt) * terms["ps"]["total"].abs() / state.ps.abs().clamp_min(1.0)).max()
    return out

def mesh_health_terms(self, mesh) -> dict[str, torch.Tensor]:
    """Return canonical mesh/operator metadata health terms."""
    corner_err = (mesh.thuburn_corner_fraction_sum - 1.0).abs()
    return {
        "cell_area_min": mesh.cell_area.min(),
        "cell_area_max": mesh.cell_area.max(),
        "cell_area_mean": mesh.cell_area.mean(),
        "dual_area_min": mesh.dual_area.min(),
        "dual_area_max": mesh.dual_area.max(),
        "dual_area_mean": mesh.dual_area.mean(),
        "dc_edge_min": mesh.dc_edge.min(),
        "dc_edge_max": mesh.dc_edge.max(),
        "dc_edge_mean": mesh.dc_edge.mean(),
        "dv_edge_min": mesh.dv_edge.min(),
        "dv_edge_max": mesh.dv_edge.max(),
        "dv_edge_mean": mesh.dv_edge.mean(),
        "corner_partition_max_abs_err": corner_err.max(),
        "corner_partition_mean_abs_err": corner_err.mean(),
        "rotation_skew_defect_max": mesh.rotation_skew_defect_max.reshape(()),
    }

def operator_identity_terms(self, mesh) -> dict[str, torch.Tensor]:
    """Return deterministic mesh-operator identity checks."""
    dtype = mesh.cell_area.dtype
    device = mesh.cell_area.device
    phi = torch.linspace(-1.0, 1.0, mesh.nCells, dtype=dtype, device=device)[:, None]
    psi = torch.linspace(0.5, -0.5, mesh.nEdges, dtype=dtype, device=device)[:, None]
    grad_phi = mesh.grad(phi)
    div_psi = mesh.div(psi)
    lhs = (grad_phi * psi * mesh.dc_edge[:, None] * mesh.dv_edge[:, None]).sum()
    rhs = -(phi * div_psi * mesh.cell_area[:, None]).sum()
    grad_div_rel = (lhs - rhs).abs() / lhs.abs().clamp_min(1.0)
    curl_grad = mesh.curl(grad_phi).abs().max()
    total_curl = mesh.integrate_dual(mesh.curl(psi)).abs().max()
    return {
        "grad_div_adjoint_relative": grad_div_rel,
        "curl_grad_max_abs": curl_grad,
        "sphere_total_curl_abs": total_curl,
    }

__all__ = ['_zero_state_like', '_state_with_tendency', '_directional_energy_rate', 'edge_metric', 'pressure_fields', 'thermo_fields', 'wind_fields', 'energy_terms', 'tracer_terms', 'static_stability', 'static_instability_penalty', 'vertical_structure_terms', 'mass_flux', 'rotated_mass_flux', 'rotation_term', 'pgf_term', 'bernoulli_term', 'mass_transport_term', 'mtheta_transport_term', 'tracer_transport_term', 'coordinate_work_term', 'tendency_terms', 'terms_total', 'force_terms', 'force_balance_terms', 'rotation_work_terms', '_edge_extensive_to_cell_layer', '_pressure_work_column_residual_stats', 'pressure_work_local_terms', 'pressure_work_local_attribution_terms', 'pressure_work_edge_exchange_terms', 'pressure_work_tangential_stencil_terms', 'pressure_work_residual_structure_terms', 'energy_exchange_terms', 'pressure_work_terms', 'cfl_terms', 'tracer_no_feedback_terms', 'divergence', 'vorticity', 'omega', 'tendency_scale_terms', 'adjustment_shock_terms', 'mesh_health_terms', 'operator_identity_terms']
