"""Snapshot A/B checks for drag, shelf transport, rest, tilt, and Hamiltonian work."""

import argparse
import json

import torch

from vfs.ocean.temperature.shelf_transport import SplitFluxGraph

from .operators import edge_drag, use_openings
from .replay import load_restart


def drag_replay(cp, initial, name, dt=600, steps=12):
    state = initial
    idx = int(state.ice.velocity.abs().argmax())
    active = cp.ridge.active_edges(initial)
    mi = cp.hamiltonian.metric(initial)
    mo = cp.ocean.energy.base.mechanical.metric(initial.ocean.ocean)[:, :1]
    mass = cp.ocean.energy.mass_area(initial.ocean.ocean)
    max_energy = max_momentum = max_stored = 0.0
    max_growth = 0.0
    history = []
    drag = (
        cp.drag_step
        if name == "baseline"
        else lambda st, sec: edge_drag(cp, st, sec, normal_only=name == "normal")
    )
    for step in range(steps):
        before = state
        ui = before.ice.velocity
        uo = before.ocean.ocean.u_e[:, :1]
        state, report = drag(before, dt)
        after_i = state.ice.velocity
        after_o = state.ocean.ocean.u_e[:, :1]
        growth = (after_i - after_o).abs() - (ui - uo).abs()
        max_growth = max(max_growth, growth[active].max().item())
        dpi = mi * (after_i - ui)
        dpo = mo * (after_o - uo)
        max_momentum = max(
            max_momentum,
            (
                (dpi + dpo).abs().max()
                / torch.maximum(dpi.abs().max(), dpo.abs().max())
            ).item(),
        )
        heat = (mass * (state.ocean.tke_h - before.ocean.tke_h)).sum()
        work = (0.5 * (after_i + ui) * dpi + 0.5 * (after_o + uo) * dpo).sum()
        max_energy = max(max_energy, ((work + heat) / heat).abs().item())
        max_stored = max(
            max_stored, ((heat - report.tke_gain.sum()) / heat).abs().item()
        )
        if name != "baseline":
            assert (report.tke_gain >= 0).all()
            assert torch.equal(after_i[~active], ui[~active]) and torch.equal(
                after_o[~active], uo[~active]
            )
            assert (growth[active] < 2e-14).all()
            assert (((after_i - after_o) * (ui - uo))[active] >= -1e-16).all()
        if step + 1 in (1, 6, 12):
            history.append(
                [
                    dt * (step + 1),
                    after_i[idx].item(),
                    (after_i - after_o)[idx].abs().item(),
                ]
            )
    assert max_energy < 3e-12 and max_momentum < 5e-11 and max_stored < 3e-12
    print(
        json.dumps(
            {
                "test": "isolated_drag",
                "mode": name,
                "edge": idx,
                "history": history,
                "max_active_slip_growth": max_growth,
                "relative_momentum": max_momentum,
                "relative_ke_tke": max_energy,
                "relative_heat": max_stored,
            }
        ),
        flush=True,
    )


def ocean_checks(owner, st):
    model = owner.ocean
    p, _ = owner.packed(st.ocean)
    s = st.ocean.ocean
    mesh, column = s.mesh, s.column
    domain = model.ocean.domain
    e, k = divmod(s.u_e.abs().argmax().item(), s.h.shape[1])
    c = mesh.cellsOnEdge[e]
    vertices = mesh.verticesOnEdge[e]
    around = mesh.edgesOnVertex[vertices[0]]
    direction = mesh.cell_xyz[c[1]] - mesh.cell_xyz[c[0]]
    tilt = 0.1 * (mesh.cell_xyz @ (direction / direction.norm()))
    h = column.layer_reference
    cases = {"saved": p}
    for key in ("rest", "stratified", "tilt"):
        hr = h.clone()
        if key == "tilt":
            hr[:, 0] += tilt
        upper = h.cumsum(1) - h
        temp = (
            2.0
            + 18.0
            * torch.exp(-upper / 250.0)
            * (-torch.expm1(-h / 250))
            * 250.0
            / torch.where(h > 0, h, 1.0)
            if key == "stratified"
            else torch.full_like(h, 10.0)
        )
        sal = torch.full_like(h, 35.0)
        z = model.state(
            u_e=torch.zeros_like(s.u_e),
            h=hr,
            conservative_temperature=temp,
            absolute_salinity=sal,
            tke=0.0,
            eddy_energy=0.1,
            tracer=torch.ones_like(hr)[..., None],
        )
        cases[key], _ = owner.packed(z)
    for mode in ("baseline", "opening"):
        if mode == "opening":
            energy = use_openings(model)
        for key, state in cases.items():
            rhs, r = model.dynamics.tendency_with_report(state, 600.0)
            eta = column.eta(state.h)
            pressure = model.dynamics.loading_acceleration(
                state, column.gravity * column.rho0 * eta, r
            )
            if key == "saved":
                if mode == "baseline":
                    original = rhs
                else:
                    for field in ("h", "temp_h", "salt_h", "tracer_h"):
                        assert torch.equal(
                            getattr(original, field), getattr(rhs, field)
                        )
                    actual = (
                        torch.zeros_like(state.u_e)
                        .flatten()
                        .index_add(0, r.faces.velocity_slot, r.faces.area)
                    )
                    widths = actual.reshape_as(state.u_e) / mesh.dv_edge[:, None]
                    opening = energy.geometry(state)[0]
                    torch.testing.assert_close(widths, opening, atol=2e-12, rtol=1e-12)
                    # Independent AD of kinetic energy using the actual transport segment areas.
                    with torch.enable_grad():
                        ah = state.h.detach().clone().requires_grad_(True)
                        az = state.with_tensors(
                            state.u_e, ah, state.temp_h, state.salt_h, state.tracer_h
                        )
                        f = SplitFluxGraph.build(az, domain)
                        ke = (
                            0.5
                            * column.rho0
                            * mesh.dc_edge[f.velocity_slot // ah.shape[1]]
                            * az.u_e.flatten()[f.velocity_slot].square()
                            * f.area
                        ).sum()
                        (g,) = torch.autograd.grad(ke, ah)
                    analytic = energy.kinetic_thickness_gradient(state)
                    free = domain.wet_cell[:, None] & column.active_layers
                    rel = ((analytic - g)[free].norm() / g[free].norm()).item()
                    assert rel < 1e-9, rel
                    print(
                        json.dumps(
                            {
                                "test": "opening_gradient",
                                "error": rel,
                                "max_opening_error_m": (widths - opening)
                                .abs()
                                .max()
                                .item(),
                            }
                        )
                    )
                grads = model.ocean.energy.base.gradients(state)
                power = [
                    grads[f] * getattr(rhs, f) for f in ("u_e", "h", "temp_h", "salt_h")
                ]
                error = (
                    sum(x.sum() for x in power) / sum(x.abs().sum() for x in power)
                ).item()
                assert abs(error) < 1e-13
            else:
                error = None
            curl = mesh.curl(pressure)[vertices, k]
            rec = {
                "test": key,
                "mode": mode,
                "edge": e,
                "level": k,
                "pressure_curl": curl.tolist(),
                "relative_work": error,
                "global_max_accel": rhs.u_e.abs().max().item(),
                "local_accel": rhs.u_e[e, k].item(),
            }
            if key == "tilt":
                ideal = -column.gravity * mesh.grad(eta)
                rec["fraction_geometric_gradient"] = (
                    pressure[around, k] / ideal[around, 0]
                ).tolist()
                if mode == "opening":
                    torch.testing.assert_close(
                        pressure[domain.active_edge_layers],
                        ideal.expand_as(pressure)[domain.active_edge_layers],
                        atol=2e-17,
                        rtol=2e-11,
                    )
            if key == "rest":
                assert rhs.u_e.abs().max() < 1e-14
            print(json.dumps(rec), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("restart", nargs="+")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    mesh = None
    with torch.no_grad():
        for path in args.restart:
            print(json.dumps({"restart": path}), flush=True)
            cp, st = load_restart(path, mesh)
            mesh = st.ocean.ocean.mesh
            for name in ("baseline", "normal", "edge"):
                drag_replay(cp, st, name)
            ocean_checks(cp, st)


if __name__ == "__main__":
    main()
