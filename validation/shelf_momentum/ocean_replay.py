"""Unforced ocean dynamics from a saved state; isolate geometry from the ice response."""

import argparse
import json

import torch

from .operators import use_openings
from .replay import load_restart


def step(model, state, dt):
    # The same four half-Euler stages and convex combination as the ocean–ice SSPRK(4,3).
    def advance(x):
        rhs, _ = model.dynamics.tendency_with_report(x, dt / 2)
        return x + (dt / 2) * rhs

    return advance((2 / 3) * state + (1 / 3) * advance(advance(advance(state))))


def energy_components(model, s):
    base = model.ocean.energy.base
    terms = base.terms(s)
    # The Roquet reference enthalpy is linear in h*CT, separately conserved.
    return {k: v.item() for k, v in terms.items() if k != "total"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("restart")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--dt", type=int, default=1200)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "opening"],
        choices=["baseline", "opening"],
    )
    args = parser.parse_args()
    if args.dt < 1 or 3600 % args.dt:
        parser.error("dt must divide one hour into integer steps")
    if args.hours < 1:
        parser.error("choose at least one hour")
    torch.set_num_threads(3)
    torch.set_default_dtype(torch.float64)
    with torch.no_grad():
        mesh = None
        for mode in args.variants:
            owner, st = load_restart(args.restart, mesh)
            model = owner.ocean
            p, _ = owner.packed(st.ocean)
            mesh = p.mesh
            if mode == "opening":
                use_openings(model)
            e, k = divmod(p.u_e.abs().argmax().item(), p.h.shape[1])
            v = mesh.verticesOnEdge[e]
            initial = energy_components(model, p)
            domain = model.ocean.domain
            wet = domain.wet_cell
            A = mesh.cell_area[wet, None]

            def stores(x, A=A, wet=wet):
                return torch.stack(
                    [
                        (A * getattr(x, field)[wet]).sum()
                        for field in ("h", "salt_h", "temp_h")
                    ]
                )

            stores0 = stores(p)
            nsteps = round(args.hours * 3600 / args.dt)
            for j in range(nsteps + 1):
                if j % max(1, round(3600 / args.dt)) == 0 or j == nsteps:
                    terms = energy_components(model, p)
                    record = {
                        "mode": mode,
                        "hours": j * args.dt / 3600,
                        "global_max_u": p.u_e.abs().max().item(),
                        "peak_location": [
                            int(q)
                            for q in torch.unravel_index(
                                p.u_e.abs().argmax(), p.u_e.shape
                            )
                        ],
                        "u": p.u_e[e, k].item(),
                        "zeta": mesh.curl(p.u_e)[v, k].tolist(),
                        "resolved_energy_change_J_m2": sum(
                            terms[n] - initial[n] for n in terms
                        )
                        / mesh.cell_area.sum().item(),
                        "global_store_relative": (
                            (stores(p) - stores0) / stores0.abs()
                        ).tolist(),
                    }
                    print(json.dumps(record), flush=True)
                if j < nsteps:
                    p = step(model, p, args.dt)


if __name__ == "__main__":
    main()
