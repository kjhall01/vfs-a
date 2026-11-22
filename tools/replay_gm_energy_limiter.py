"""One-campaign exploratory GM *flux-level* energy admissibility replay.

This is a research experiment, NOT a general production correction.  It only
retries a GM *funding* failure, locally shrinking the adjacent bolus
streamfunctions. On every retry, it evaluates the original GMRedi transport,
Hamiltonian work, energy flux, and admissibility guards from the unchanged
start-of-substep state. Neither the prognostic reservoir nor the other state
variables are clipped, floored, or replaced; Redi failures remain failures.

Put this under tools/, outside the coupled campaign's source hash roots, so
its original pinned checkpoint remains directly restartable. Run with the
usual campaign.py flags, --resume, --output, and the same configuration.

The output writes gm_limiter_events.jsonl for any corrected GM substeps. The
model's own diagnostics, global energy ledger, and all failure guards remain.
"""

import argparse
import json
from pathlib import Path
import sys

import torch


def incident_gm_interfaces(bad, i, j):
    """For bad cell/layer k, only adjacent interfaces k and k+1 can affect it.

    A bolus psi at interface k changes horizontal transport in layers k-1 and
    k and vertical transport between those layers. The i,j indices refer to
    active horizontal edges, not all mesh edges, so coastal masks are intact.
    """
    touching_layers = bad[i] | bad[j]  # active edge x layer
    return touching_layers[:, :-1] | touching_layers[:, 1:]


def limited_gm_fluxes(psi, factor, metrics, h, divergence):
    """Rebuild *both* bolus carriers from limited interface streamfunctions.

    Never scale horizontal transport alone: the vertical carrier must be
    recomputed to retain the paired three-dimensional divergence-free flow.
    """
    z, vertical_ids, vertical_weights, i, j, dc, ell, ids, zn = metrics
    if factor.shape != psi[:, 1:-1].shape:
        raise ValueError("GM face limiter shape mismatch")
    psi = torch.cat((psi[:, :1], psi[:, 1:-1] * factor, psi[:, -1:]), dim=1)
    horizontal = ell * psi.diff(dim=1)
    hi, hj = ids[i].flatten(), ids[j].flatten()
    hdiv = divergence(hi, hj, horizontal.flatten(), h.numel()).reshape_as(h)
    vertical = torch.cat((
        torch.zeros_like(h[:, :1]),
        hdiv.cumsum(dim=1)[:, :-1],
        torch.zeros_like(h[:, :1]),
    ), dim=1)
    vi, vj = ids[:, :-1].flatten(), ids[:, 1:].flatten()
    ii, jj = torch.cat((hi, vi)), torch.cat((hj, vj))
    carrier = torch.cat((horizontal.flatten(), vertical[:, 1:-1].flatten()))
    return psi, horizontal, vertical, ii, jj, carrier


def make_experimental_gm_class(GMRedi, divergence, log_path):
    class UnfundedGM(FloatingPointError):
        def __init__(self, bad, worst_deficit):
            super().__init__("GM requires unavailable local eddy energy")
            self.bad = bad
            self.worst_deficit = worst_deficit

    class BudgetedGMRedi(GMRedi):
        """Unchanged baseline fast path; local flux limiting only on failure."""
        max_passes = 20
        per_pass_factor = 0.5

        def _geometry(self, model, state, *, part="all", metrics=None,
                      gm_diffusivity=None):
            geom = super()._geometry(model, state, part=part, metrics=metrics,
                                     gm_diffusivity=gm_diffusivity)
            if part != "gm":
                return geom
            if metrics is None:
                metrics = self._metrics(model, state)
            self._active_i, self._active_j = metrics[3], metrics[4]
            if self._factor is None:
                return geom  # bitwise original code until the first funding failure
            slope, psi, horizontal, vertical, ii, jj, carrier, operator, measure = geom
            psi, horizontal, vertical, ii, jj, carrier = limited_gm_fluxes(
                psi, self._factor, metrics, state.ocean.h, divergence)
            return slope, psi, horizontal, vertical, ii, jj, carrier, operator, measure

        @staticmethod
        def _settle(model, before, transported, energy_inflow, name):
            try:
                return GMRedi._settle(model, before, transported,
                                      energy_inflow, name)
            except FloatingPointError as original:
                if name != "GM" or "unavailable local eddy energy" not in str(original):
                    raise
                # The original _settle already failed; inspect its exact finite
                # work and no other budget/physics formula.
                H = model.hamiltonian
                work = H.finite_change(before, transported) - energy_inflow
                mass = H.mass_area(before.ocean)
                safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
                resulting_eddy_h = transported.eddy_h - work / safe_mass
                bad = resulting_eddy_h < 0
                if not bool(bad.any()):
                    raise original
                safe_h = torch.where(before.ocean.h > 0, before.ocean.h, torch.ones_like(before.ocean.h))
                per_kg_deficit = (-resulting_eddy_h / safe_h)[bad]
                raise UnfundedGM(bad, float(per_kg_deficit.max())) from original

        def apply(self, model, state, dt):
            self._factor = None
            self._active_i = self._active_j = None
            first_deficit = None
            first_bad_count = None
            passes = 0
            try:
                for attempt in range(self.max_passes + 1):
                    try:
                        end, report = super().apply(model, state, dt)
                        if passes:
                            entry = {
                                "dt_seconds": float(dt),
                                "gm_budget_attempts": attempt + 1,
                                "limiting_passes": passes,
                                "initial_bad_cell_layers": first_bad_count,
                                "initial_worst_deficit_J_kg": first_deficit,
                                "locally_scaled_interfaces": int((self._factor < 1).sum()),
                                "minimum_streamfunction_factor": float(self._factor.min()),
                                "final_min_eddy_J_kg": float((end.eddy_h / end.ocean.h).min()),
                                "max_GM_volume_divergence": float(report.volume_residual.abs().max()),
                                "GM_local_energy_residual_max_J": float(report.gm_local_energy_residual.abs().max()),
                            }
                            log_path.parent.mkdir(parents=True, exist_ok=True)
                            with log_path.open("a") as out:
                                out.write(json.dumps(entry) + "\n")
                            print("LOCAL GM ENERGY LIMITER: " + json.dumps(entry), flush=True)
                        return end, report
                    except UnfundedGM as error:
                        if first_deficit is None:
                            first_deficit = error.worst_deficit
                            first_bad_count = int(error.bad.sum())
                        new_mask = incident_gm_interfaces(
                            error.bad, self._active_i, self._active_j)
                        if not bool(new_mask.any()):
                            raise FloatingPointError(
                                "GM local funding failed but no active incident interfaces exist"
                            ) from error
                        if self._factor is None:
                            self._factor = torch.ones(
                                new_mask.shape, dtype=state.ocean.h.dtype,
                                device=state.ocean.h.device)
                        self._factor = torch.where(new_mask,
                                                   self._factor * self.per_pass_factor,
                                                   self._factor)
                        passes += 1
                raise FloatingPointError(
                    "GM local energy limiter exhausted 20 passes without admissible transport"
                )
            finally:
                self._factor = None
                self._active_i = self._active_j = None

    return BudgetedGMRedi


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path, required=True)
    args, _ = p.parse_known_args()
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from workflows.coupled import campaign
    from eddy import GMRedi
    from vfs.ocean.transport import divergence

    campaign.GMRedi = make_experimental_gm_class(
        GMRedi, divergence, args.output / "gm_limiter_events.jsonl"
    )
    campaign.main()


if __name__ == "__main__":
    main()