"""Replay a coupled checkpoint and report the FIRST failed local GM/Redi energy settlement.

This script changes no physical state, discretization, or numerical tolerance.
It temporarily selects an observational subclass only in this process. Because
``tools/`` is not included in the coupled source-fingerprint inventory, a
checkpoint created by the same physics commit remains directly restartable.

Invocation: use exactly the same options as workflows/coupled/campaign.py,
including --resume, --output (new/empty), and --stop-after-days 9.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflows.coupled import campaign
from eddy import GMRedi


class ReportingGMRedi(GMRedi):
    """Call the unchanged settlement; inspect tensors only after it rejects."""

    diagnostic_path = None
    invocation_count = 0

    def apply(self, model, state, dt):
        type(self).invocation_count += 1
        return super().apply(model, state, dt)

    @staticmethod
    def _settle(model, before, transported, energy_inflow, name):
        try:
            return GMRedi._settle(model, before, transported, energy_inflow, name)
        except FloatingPointError as exc:
            if "requires unavailable local eddy energy" not in str(exc):
                raise
            with torch.no_grad():
                H = model.hamiltonian
                resolved_change = H.finite_change(before, transported)
                work = resolved_change - energy_inflow
                mass = H.mass_area(before.ocean)
                safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
                h = before.ocean.h
                specific_transport = transported.eddy_h / h
                specific_debit = work / safe_mass / h
                specific_after = specific_transport - specific_debit
                invalid = specific_after < 0
                bad_idx = torch.nonzero(invalid.reshape(-1), as_tuple=False).reshape(-1)
                nlevels = h.shape[-1]
                values = specific_after.reshape(-1)[bad_idx]
                bad_sorted = bad_idx[torch.argsort(values)[:10]]
                xyz = before.ocean.mesh.cell_xyz

                def scalar(t, c, z):
                    return float(t[c, z].detach().cpu())

                rows = []
                for j in bad_sorted.tolist():
                    c, z = divmod(j, nlevels)
                    p = xyz[c].detach().cpu()
                    rows.append(dict(
                        cell=c, layer=z,
                        latitude_deg=math.degrees(math.asin(float(p[2]))),
                        longitude_deg=math.degrees(math.atan2(float(p[1]), float(p[0]))),
                        initial_e_J_kg=scalar(before.eddy_h / h, c, z),
                        transported_e_J_kg=scalar(specific_transport, c, z),
                        required_conversion_J_kg=scalar(specific_debit, c, z),
                        deficit_J_kg=-scalar(specific_after, c, z),
                        resolved_change_J=scalar(resolved_change, c, z),
                        energy_inflow_J=scalar(energy_inflow, c, z),
                        thickness_m=scalar(h, c, z),
                    ))
                info = dict(
                    stage=name,
                    error=str(exc),
                    gm_redi_apply_number=ReportingGMRedi.invocation_count,
                    negative_cell_layers=int(bad_idx.numel()),
                    min_remaining_e_J_kg=float(specific_after.min()),
                    min_transported_e_J_kg=float(specific_transport.min()),
                    max_required_conversion_J_kg=float(specific_debit.max()),
                    min_available_before_J_kg=float((before.eddy_h / h).min()),
                    min_available_after_transport_J_kg=float(specific_transport.min()),
                    worst_cell_layers=rows,
                )
                path = ReportingGMRedi.diagnostic_path
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(info, indent=2) + "\n")
                print("GM FUNDING DIAGNOSTIC: " + json.dumps(info, indent=2), flush=True)
            raise


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    args, _ = parser.parse_known_args()
    ReportingGMRedi.diagnostic_path = args.output / "gm_budget_failure.json"
    # Only the campaign's GMRedi constructor is redirected, to the subclass
    # which delegates every successful operation to the original implementation.
    campaign.GMRedi = ReportingGMRedi
    campaign.main()


if __name__ == "__main__":
    main()