"""Report a failed local GM/Redi energy settlement using the ordinary campaign arguments.

The diagnostic records cell-layer budgets only when the operator rejects a
complete GM/Redi call. Successful calls use the native results, including any
funding retries of the configured operator.
"""
import argparse
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows.coupled import campaign


@torch.no_grad()
def failure_report(model, before, transported, energy_inflow, name, error, call):
    H = model.hamiltonian
    resolved_change = H.finite_change(before, transported)
    work = resolved_change - energy_inflow
    mass = H.mass_area(before.ocean)
    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
    h = before.ocean.h
    specific_transport = transported.eddy_h / h
    specific_debit = work / safe_mass / h
    # Match the order of operations in the native h*e settlement.
    remaining = transported.eddy_h - work / safe_mass
    specific_after = remaining / h
    invalid = remaining < 0
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
        error=str(error),
        gm_redi_apply_number=call,
        negative_cell_layers=int(bad_idx.numel()),
        min_remaining_e_J_kg=float(specific_after.min()),
        min_transported_e_J_kg=float(specific_transport.min()),
        max_required_conversion_J_kg=float(specific_debit.max()),
        min_available_before_J_kg=float((before.eddy_h / h).min()),
        worst_cell_layers=rows,
    )
    return info


def reporting_operator(native, report_path):
    call = 0

    class ReportingGMRedi(native):
        def apply(self, model, state, dt):
            nonlocal call
            call += 1
            try:
                return super().apply(model, state, dt)
            except FloatingPointError as error:
                # The funding limiter can chain an exhaustion error from the
                # last local rejection. Ignore earlier rejections it recovered.
                cause = error
                while cause is not None:
                    settlement = getattr(cause, "gm_settlement", None)
                    if settlement is not None:
                        campaign.atomic_json(report_path, failure_report(*settlement, error, call))
                        break
                    cause = cause.__cause__
                raise

        @staticmethod
        def _settle(model, before, transported, energy_inflow, name):
            try:
                return native._settle(model, before, transported, energy_inflow, name)
            except FloatingPointError as error:
                if "requires unavailable local eddy energy" in str(error):
                    error.gm_settlement = model, before, transported, energy_inflow, name
                raise

    return ReportingGMRedi


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    operator = reporting_operator(campaign.GMRedi, args.output / "gm_budget_failure.json")
    with patch.object(campaign, "GMRedi", operator):
        return campaign.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
