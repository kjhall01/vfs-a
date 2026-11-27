"""Measure how the existing ALE energy ledger behaves under the candidate metric."""

import argparse
import json
from types import SimpleNamespace

import torch

from workflows.coupled.campaign import CampaignConfig, ale_step

from .operators import use_openings
from .replay import load_restart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("restart")
    parser.add_argument("manifest")
    args = parser.parse_args()
    with open(args.manifest) as stream:
        config = CampaignConfig(**json.load(stream)["configuration"])
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    with torch.no_grad():
        mesh = None
        for mode in ("baseline", "opening_old_ALE", "opening_matched_ALE"):
            cp, state = load_restart(args.restart, mesh)
            model = cp.ocean
            if mode != "baseline":
                use_openings(model, match_ale=mode == "opening_matched_ALE")
            s = state.ocean.ocean
            mesh = s.mesh
            A = mesh.cell_area.sum()
            mech = model.ocean.energy.base.mechanical

            def ke_offset(x, mech=mech, A=A):
                direct = 0.5 * (mech.metric(x) * x.u_e.square()).sum()
                attributed = mech.column_kinetic_energy(x).sum()
                return ((direct - attributed) / A).item()

            before = ke_offset(s)
            # ALE settles its local column ledger, while the outer campaign reports the global Hamiltonian.
            from workflows.coupled.ice_temperature import ThermalIceState

            sea = ThermalIceState(
                state,
                torch.zeros_like(state.ice.area),
                torch.zeros_like(state.ice.area),
            )
            record = {"mode": mode, "ke_minus_column_ledger_before_J_m2": before}
            try:
                end, r = ale_step(SimpleNamespace(H=cp.hamiltonian), sea, config)
                final = end.coupled.ocean.ocean
                after = ke_offset(final)
                mi0, mi1 = mech.metric(s), mech.metric(final)
                momentum0, momentum1 = mi0 * s.u_e, mi1 * final.u_e
                momentum_error = (momentum1 - momentum0).sum(
                    1
                ).abs().max() / momentum0.abs().sum(1).max()
                energy0 = model.hamiltonian.terms(state.ocean)
                energy1 = model.hamiltonian.terms(end.coupled.ocean)
                actual = (
                    sum(
                        (energy1[n] - v).item()
                        for n, v in energy0.items()
                        if n != "total"
                    )
                    / A.item()
                )
                record.update(
                    changed=r["ale_changed_columns"],
                    deferred=r["ale_deferred_columns"],
                    reported_change_J_m2=r["ale_energy_change_J_m2"],
                    ke_minus_column_ledger_after_J_m2=after,
                    actual_change_J_m2=actual,
                    relative_column_edge_momentum_error=momentum_error.item(),
                    column_ledger_change_J_m2=actual - (after - before),
                    max_reported_column_error_J_m2=r["ale_column_residual_J_m2"],
                )
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                record["rejection"] = str(exc)
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
