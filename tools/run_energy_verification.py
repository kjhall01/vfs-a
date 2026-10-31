"""Increment 1b: small C2/C3/C4 measurements on the unchanged installed core.

All autograd and finite differences are independent verification diagnostics.
Production tendencies and stages run normally; only serialized artifacts detach.
"""
import argparse
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
from unittest.mock import patch

import torch
from vfs import (DryAtmosphere, MoistAtmosphere, State, VoronoiMesh,
                 CharneyPhillipsColumn, make_hybrid_coefficients)
from vfs.atmosphere.bounded_transport import graph
from vfs.atmosphere.moist_budgets import inventories

spec = importlib.util.spec_from_file_location("verification_common", Path(__file__).with_name("run_verification.py"))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
MOIST = ("u_e", "ps", "entropy_content", "water_content")
DRY = ("u_e", "ps", "mtheta_i")
ENERGIES = ("kinetic", "internal", "potential", "boundary", "total")


def fields(state):
    return MOIST if hasattr(state, "water_content") else DRY


def payload(state):
    return {key: getattr(state, key) for key in fields(state)}


def state_with(state, **values):
    """Dry State owns a payload dataclass; moist state exposes its tensors directly."""
    if fields(state) == MOIST:
        return replace(state, **values)
    return State(**dict(payload(state), **values), mesh=state.mesh, column=state.column)


def archive(value):
    """Detach only at the output boundary, never in a diagnostic computation."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: archive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [archive(item) for item in value]
    return value


def components(model, state):
    result = model.dynamics.energy_terms(state)
    if fields(state) == DRY:
        result = dict(result, boundary=model.dynamics.boundary_reservoir(state),
                      total=model.dynamics.boundary_total_energy(state))
    return {key: result[key] for key in ENERGIES}


def energy_derivatives(model, state):
    """Differentiate scalar energies, independently of analytic gradient code."""
    keys = fields(state)
    leaves = [getattr(state, key).clone().requires_grad_() for key in keys]
    probe = state_with(state, **dict(zip(keys, leaves)))
    energies = components(model, probe)
    gradients = {}
    for name, energy in energies.items():
        grads = torch.autograd.grad(energy, leaves, retain_graph=True, allow_unused=True)
        gradients[name] = {key: torch.zeros_like(leaf) if grad is None else grad
                           for key, leaf, grad in zip(keys, leaves, grads)}
    return energies, gradients


def analytic_work(model, state, direction):
    if fields(state) == MOIST:
        return model.dynamics.boundary_energy_rate(state, direction)
    return model.dynamics.boundary_energy_rate_closed(state, direction)


def make_case(kind, cells, levels, seed):
    """Diagnostic atmosphere, not a balanced-flow or terrain-rest benchmark.

    Small smooth terrain phi_s=1000*x m²/s² and varying ps prevent a trivial
    uniform-state derivative test. Sharp cases use exactly absent condensate
    regions to exercise limiting; C2 uses only strictly positive smooth cases.
    """
    mesh = VoronoiMesh(n_cells=cells, lloyd_iters=0, radius=6371220., verbose=False)
    a, b = make_hybrid_coefficients(levels, p_top=5000.)
    x = mesh.cell_xyz[:, :1]
    column = CharneyPhillipsColumn(levels, a, b, phi_s=1000*x[:, 0],
                                  phi_full_mode="direct_u_gl2_adjoint")
    ps = 100000*(1 + .01*x)
    generator = torch.Generator().manual_seed(seed)
    wind = 10*torch.randn(mesh.nEdges, levels, generator=generator)
    if kind == "dry":
        model = DryAtmosphere(mesh, column).to(dtype=torch.float64)
        state = State.from_theta(wind, 300 + 50*(1-column.B_interface)[None, :] + 2*x, ps,
                                 mesh=mesh, column=column)
    else:
        model = MoistAtmosphere(mesh, column).to(dtype=torch.float64)
        pressure = column.interface_pressure(ps)
        liquid = .001*(1+.1*x) if kind == "smooth" else .002*(x > 0)*(pressure/100000)**2
        ice = .0001*(1+.1*x) if kind == "smooth" else 0.
        state = model.state_from_fields(
            temperature=280*(pressure/100000)**.08*(1+.01*x),
            surface_pressure=ps, normal_wind=-wind if kind == "reversed" else wind,
            vapor=.01*(1+.1*x), liquid=liquid, ice=ice)
    return model, state


def derivative_sweeps(model, state, epsilons, directions, seed):
    """E(x±epsilon*d), epsilon dimensionless; d has native state units.

    Each species has its own direction scale. All other prognostic entries are
    held fixed, including entropy when composition changes. Report derivative
    errors in J per unit epsilon and unnormalized Taylor remainders in J.
    """
    energies, gradients = energy_derivatives(model, state)
    generator = torch.Generator().manual_seed(seed)
    blocks = [(key, None) for key in fields(state) if key != "water_content"]
    if fields(state) == MOIST:
        blocks += [("water_content", species) for species in range(3)]
    rows, saved = [], {}
    for key, species in blocks:
        value = getattr(state, key) if species is None else state.water_content[..., species]
        scale = 10. if key == "u_e" else 1000. if key == "ps" else float(value.abs().mean())*.1
        for number in range(directions):
            delta = torch.randn(value.shape, generator=generator)
            delta = delta / delta.square().mean().sqrt() * scale
            direction = {name: torch.zeros_like(tensor) for name, tensor in payload(state).items()}
            if species is None:
                direction[key] = delta
            else:
                direction[key][..., species] = delta
            label = f"{key}:{species}:{number}"
            saved[label] = direction
            analytic = analytic_work(model, state, state_with(state, **direction))
            ad = {name: sum((grad[k]*direction[k]).sum() for k in fields(state))
                  for name, grad in gradients.items()}
            for epsilon in epsilons:
                plus = state_with(state, **{k: v + epsilon*direction[k] for k, v in payload(state).items()})
                minus = state_with(state, **{k: v - epsilon*direction[k] for k, v in payload(state).items()})
                # Production validation is retained: no clipping or perturbation retry.
                ep, em = components(model, plus), components(model, minus)
                for name in ENERGIES:
                    fd = (ep[name]-em[name])/(2*epsilon)
                    rows.append(dict(direction=label, scale_native=scale, epsilon=epsilon,
                        energy=name, ad_J=float(ad[name]), fd_J=float(fd),
                        fd_minus_ad_J=float(fd-ad[name]),
                        relative_error=common.normalized_residual(float(abs(fd-ad[name])), float(abs(ad[name]))),
                        forward_taylor_remainder_J=float(ep[name]-energies[name]-epsilon*ad[name]),
                        backward_taylor_remainder_J=float(em[name]-energies[name]+epsilon*ad[name]),
                        analytic_total_J=float(analytic) if name == "total" else None,
                        analytic_minus_ad_J=float(analytic-ad[name]) if name == "total" else None))
    return rows, saved


def work_closure(model, state, dt):
    """Resolve base component work and bounded-minus-centered changes in W.

    Do not sum aliases such as advective/total twice. Internal, potential,
    kinetic and fixed-lid energy gradients are obtained independently with AD.
    Production total-energy contraction is reported alongside the AD contraction.
    """
    energies, gradients = energy_derivatives(model, state)
    terms = model.dynamics.tendency_terms(state)
    base = model.dynamics.tendency(state)
    scalar = "entropy_content" if fields(state) == MOIST else "mtheta_i"
    selected = {"u_e": ("pgf", "rotation", "vertical", "bernoulli"),
                "ps": ("mass_flux",), scalar: ("horizontal", "vertical")}
    if fields(state) == MOIST:
        selected["water_content"] = ("total",)
    pieces = {(key, name): terms[key][name] for key, names in selected.items() for name in names}
    rhs = base
    heat = state.ps.new_zeros(())
    if fields(state) == MOIST:
        rhs, info = model.stepper.transport.evaluate(state, dt)
        pieces["u_e", "bounded_gibbs_change"] = rhs.u_e-base.u_e-info["limiter_force"]
        pieces["u_e", "limiter_mechanical"] = info["limiter_force"]
        pieces[scalar, "limited_transport_change"] = rhs.entropy_content-base.entropy_content-info["entropy_source"]
        pieces[scalar, "heat_allocation"] = info["entropy_source"]
        pieces["water_content", "limited_transport_change"] = rhs.water_content-base.water_content
        heat = info["heat"].sum()
    rows = []
    for (key, term), tendency in pieces.items():
        rows.append(dict(field=key, term=term, work_W={name: float((grad[key]*tendency).sum())
                                                      for name, grad in gradients.items()}))
    total = sum((gradients["total"][key]*getattr(rhs, key)).sum() for key in fields(state))
    analytic = analytic_work(model, state, rhs)
    cancellation = sum(abs(row["work_W"]["total"]) for row in rows)
    assembly = {key: float((sum(value for (field, _), value in pieces.items() if field == key)
                           - getattr(rhs, key)).abs().max()) for key in fields(state)}
    return dict(energy_J={name: float(value) for name, value in energies.items()},
                components=rows, ad_total_W=float(total), analytic_total_W=float(analytic),
                analytic_minus_ad_W=float(analytic-total),
                component_sum_minus_rhs_work_W=sum(row["work_W"]["total"] for row in rows)-float(total),
                assembly_max_error_native_per_s=assembly, cancellation_scale_W=cancellation,
                normalized_closure=common.normalized_residual(float(total), cancellation),
                closure_W_m2=float(total/state.mesh.cell_area.sum()), allocated_heat_W=float(heat),
                heat_work_minus_allocated_W=sum(row["work_W"]["total"] for row in rows if row["term"] == "heat_allocation")-float(heat)), pieces


def independent_amounts(state, gravity):
    """Carrier kg at CP nodes from adjacent half-layer pressure thicknesses.

    Water is prognostic pressure content times area/g; physical dry is carrier
    minus the three water amounts. This does not call the graph/ledger helper.
    """
    dp = state.column.interface_pressure(state.ps).diff()
    half = dp/2
    dm = torch.cat((half[:, :1], half[:, :-1]+half[:, 1:], half[:, -1:]), -1)
    carrier = dm*state.mesh.cell_area[:, None]/gravity
    water = state.water_content*state.mesh.cell_area[:, None, None]/gravity
    return carrier.flatten(), torch.cat(((carrier-water.sum(-1))[..., None], water), -1).reshape(-1, 4)


def incident_sum(i, j, transfers, nodes):
    """Independent face audit: sum incoming minus outgoing at each node.

    Deliberately avoid the production index_add divergence. Different reduction
    order introduces roundoff; residuals are reported rather than overwritten.
    """
    return torch.stack([transfers[j == node].sum(0)-transfers[i == node].sum(0)
                        for node in range(nodes)])


def stage_budget(base, initial, updated, info, dt, dry_floor, vapor_floor):
    gravity = base.kernel.gravity
    carrier0, species0 = independent_amounts(initial, gravity)
    carrier1, species1 = independent_amounts(updated, gravity)
    _, i, j, flux = graph(base, initial)
    amounts = info["amounts"]/gravity
    transfer = info["integrated_flux"]/gravity
    carrier_transfer = dt*flux/gravity
    carrier_residual = carrier1-carrier0-incident_sum(i, j, carrier_transfer, carrier0.numel())
    species_residual = amounts-species0-incident_sum(i, j, transfer, carrier0.numel())
    expected_water = info["amounts"][:, 1:].reshape_as(updated.water_content)/initial.mesh.cell_area[:, None, None]
    exact_ledger = torch.equal(updated.water_content, expected_water)
    if not exact_ledger:
        raise ValueError("Euler water is not the exact authoritative ledger")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in (carrier1, amounts, transfer, info["beta"])):
        raise ValueError("Nonfinite stage budget")
    if bool(((info["beta"] < 0) | (info["beta"] > 1)).any()):
        raise ValueError("Limiter factors outside [0,1]")
    if info["limited_faces"] != int((info["beta"] < 1).sum()):
        raise ValueError("Limiter count disagrees with the recorded mask")
    floors = carrier1.new_tensor([dry_floor, vapor_floor, 0., 0.])
    production = inventories(base, updated)
    result = dict(dt_s=dt, exact_water_ledger=exact_ledger,
        carrier_initial_kg=float(carrier0.sum()), carrier_final_kg=float(carrier1.sum()),
        species_initial_kg=species0.sum(0).tolist(), species_final_kg=species1.sum(0).tolist(),
        species_ledger_kg=amounts.sum(0).tolist(),
        species_global_change_kg=(species1.sum(0)-species0.sum(0)).tolist(),
        carrier_global_change_kg=float(carrier1.sum()-carrier0.sum()),
        carrier_incidence_max_error_kg=float(carrier_residual.abs().max()),
        species_incidence_max_error_kg=species_residual.abs().amax(0).tolist(),
        ledger_sum_minus_target_max_kg=float((amounts.sum(-1)-info["target_mass"]/gravity).abs().max()),
        species_flux_sum_minus_carrier_max_kg=float((transfer.sum(-1)-carrier_transfer).abs().max()),
        physical_dry_minus_ledger_max_kg=float((species1[:, 0]-amounts[:, 0]).abs().max()),
        minimum_carrier_kg=float(carrier1.min()), minimum_species_kg=species1.amin(0).tolist(),
        minimum_fraction=(species1/carrier1[:, None]).amin(0).tolist(),
        maximum_fraction=(species1/carrier1[:, None]).amax(0).tolist(),
        minimum_floor_margin_kg=(species1-carrier1[:, None]*floors).amin(0).tolist(),
        carrier_amount_scale_kg=float(carrier0.abs().max()),
        species_amount_scales_kg=species0.abs().amax(0).tolist(),
        carrier_incidence_normalized=common.normalized_residual(float(carrier_residual.abs().max()), float(carrier0.abs().max())),
        species_incidence_normalized=[common.normalized_residual(float(e), float(s)) for e, s in zip(species_residual.abs().amax(0), species0.abs().amax(0))],
        limited_faces=info["limited_faces"], donor_substeps=info["substeps"],
        beta_min=float(info["beta"].min()), beta_max=float(info["beta"].max()),
        production_mass_minus_independent_kg=float(production["mass"]-carrier1.sum()),
        production_water_minus_independent_kg=(production["species"]-species1[:, 1:].sum(0)).tolist())
    return result, dict(i=i, j=j, carrier_transfer_kg=carrier_transfer,
                        carrier_incidence_residual_kg=carrier_residual,
                        species_incidence_residual_kg=species_residual)


def record_stages(model, state, dt):
    """Observe the production Euler calls; return their unchanged results.

    No RK coefficients or stage construction are copied into this driver.
    The observation patch is local to this single-threaded verification call.
    """
    transport = model.stepper.transport
    original = transport.euler_with_info
    rows, raw = [], []
    def observe(initial, budget):
        updated, info = original(initial, budget)
        result, audit = stage_budget(model.dynamics, initial, updated, info, budget,
                                     transport.dry_floor, transport.vapor_floor)
        rows.append(result)
        raw.append(archive(dict(initial=payload(initial), euler_output=payload(updated),
                                info=info, audit=audit)))
        return updated, info
    with patch.object(transport, "euler_with_info", side_effect=observe):
        updated = model(state, dt)
    if len(rows) != 3 or any(row["dt_s"] != dt for row in rows):
        raise ValueError("Expected three full-dt Euler stage budgets")
    return updated, rows, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--numerical-revision", choices=[common.NUMERICAL_COMMIT])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--dt", type=float, default=1.)
    parser.add_argument("--directions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if args.cells < 4 or args.levels < 1 or args.dt <= 0 or args.directions < 1:
        parser.error("Positive levels/dt/directions and at least four cells required")
    epsilons = [10.**(-k) for k in range(2, 8)]
    configuration = dict(cells=args.cells, levels=args.levels, dt_s=args.dt, steps=1,
        directions=args.directions, seed=args.seed, epsilons=epsilons,
        dtype="float64", device="cpu", threads=1, radius_m=6371220., lloyd_iters=0,
        initialization="make_case: diagnostic smooth terrain, not terrain rest",
        configurations=["M-B:smooth", "M-B:sharp", "M-B:reversed", "D-E:dry"],
        stepper="BoundedSSPRK3 or DrySSPRK3 defaults", solver=None,
        phi_full_mode="direct_u_gl2_adjoint", p_top_Pa=5000.,
        cadence="initial C2/C3, every actual moist Euler stage, final state",
        differentiation="scalar diagnostic only; no timestep backward qualification")
    with common.recorded_run(args.repo, args.output, "C2-DER-001+C3-WORK-001+C4-LEDGER-001", configuration, __file__, args.numerical_revision):
        results = {}
        for kind in ("smooth", "sharp", "reversed", "dry"):
            model, state = make_case(kind, args.cells, args.levels, args.seed)
            initial = archive(payload(state))
            work, pieces = work_closure(model, state, args.dt)
            derivatives, directions = derivative_sweeps(model, state, epsilons, args.directions, args.seed) if kind in ("smooth", "dry") else ([], {})
            if kind == "dry":
                updated, stages, raw_stages = model(state, args.dt), [], []
            else:
                updated, stages, raw_stages = record_stages(model, state, args.dt)
            for key in fields(state):
                if not torch.equal(getattr(state, key), initial[key]):
                    raise ValueError(f"Input mutation in {kind}/{key}")
            energy_change = model.dynamics.boundary_total_energy(updated)-model.dynamics.boundary_total_energy(state)
            constants = asdict(model.dynamics.thermodynamics) if kind != "dry" else dict(c_p=state.column.c_p, R_d=state.column.R_d, p0=state.column.p0)
            constants["rotation_rate"] = model.dynamics.kernel.rotation_rate if kind != "dry" else model.dynamics.rotation_rate
            constants["gravity"] = model.dynamics.kernel.gravity if kind != "dry" else model.dynamics.gravity
            final_budget = {}
            if kind != "dry":
                mass0, species0 = independent_amounts(state, constants["gravity"])
                mass1, species1 = independent_amounts(updated, constants["gravity"])
                final_budget = dict(carrier_change_kg=float(mass1.sum()-mass0.sum()),
                    species_change_kg=(species1.sum(0)-species0.sum(0)).tolist(),
                    minimum_fraction=(species1/mass1[:, None]).amin(0).tolist(),
                    dry_floor=model.stepper.transport.dry_floor,
                    vapor_floor=model.stepper.transport.vapor_floor,
                    allow_condensate_roundoff=model.dynamics.allow_condensate_roundoff)
            results[kind] = dict(C2=derivatives, C3=work, C4=stages,
                                 final_convex_combination_budget=final_budget,
                                 finite_step_energy_change_J=float(energy_change), constants=constants)
            torch.save(archive(dict(initial=initial, final=payload(updated), geometry=model.state_dict(),
                directions=directions, work_tendencies={f"{k}/{n}": v for (k, n), v in pieces.items()},
                stages=raw_stages)), args.output / f"{kind}.pt")
            common.write_json(args.output / "metrics.json", results)


if __name__ == "__main__":
    main()
