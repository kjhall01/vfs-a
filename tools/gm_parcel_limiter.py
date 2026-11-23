"""Local GM streamfunction funding experiment for the donor-parcel campaign.

The transport and audit functions below reproduce the day-10 checkpoint replay.
Identities for this experiment carry this file alongside the native source
inventory. Only the run_gm_day11 launcher installs the experimental operator.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch

from run_gm_day9 import load_campaign


# Digest of the 259 hashed sources in the donor-parcel day-10 manifest.
DONOR_SOURCE_MAP_SHA256 = "d425abea9676be17ead4c54095a0dcf8a443085712bd438bbc7a964351f29ccd"
EXPERIMENT_SOURCE = "tools/gm_parcel_limiter.py"
DAY11_LIMITER_SHA256 = "b42ecb8f033637ab5f6914bbdae899de37dc02cdb259e54a35e8a732b4b89479"
# Only these native modules changed between the day-11 experiment and its
# bounded category-phase continuation. All other source entries stay pinned
# by the original donor campaign digest.
DAY11_ICE_SOURCES = {
    "validation/sea_ice/category_phase.py": "44ad80734ac1450b97a20a3a8f1b6c9c67b7369e12fd9d3ec63a04bb693b8034",
    "validation/sea_ice/ocean_ice.py": "9aa08e43fed7f3b0c77e041113f1630ea9def6d767d8923b07ba975faa54e051",
}
BOUNDED_ICE_SOURCES = {
    "validation/sea_ice/category_phase.py": "31270fef6cc80a3fa06d23d2304fc9e713b21eb181e98429b3799e8c6d3a56b6",
    "validation/sea_ice/ocean_ice.py": "48e078d70601e741a68e81a859f99059500d351ee172aa0a85c2eb3e9ac42963",
}


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


def make_audited_gm_class(GMRedi, divergence, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    limiter = make_experimental_gm_class(GMRedi, divergence, output_dir / "gm_limiter_events.jsonl")
    step = 0

    class AuditedGM(limiter):
        def _geometry(self, model, state, *, part="all", metrics=None, gm_diffusivity=None):
            geom = super()._geometry(model, state, part=part, metrics=metrics, gm_diffusivity=gm_diffusivity)
            if part == "gm" and self._factor is None:
                self._unlimited = geom[1:4]  # psi, horizontal, vertical
                self._length = metrics[6]
                self._height = metrics[0]
                self._diffusivity = gm_diffusivity
            if part == "gm":
                self._final_factor = self._factor
            return geom

        def _settle(self, model, before, transported, energy_inflow, name):
            try:
                return super()._settle(model, before, transported, energy_inflow, name)
            except FloatingPointError as exc:
                if not hasattr(exc, "bad"):
                    raise
                bad_positions = exc.bad.nonzero()
                c, k = bad_positions.unbind(1)
                available = (transported.eddy_h[c,k] / before.ocean.h[c,k]).tolist()
                work = model.hamiltonian.finite_change(before, transported)[c,k] - energy_inflow[c,k]
                mass = model.hamiltonian.mass_area(before.ocean).expand_as(before.ocean.h)
                cost = (work / mass[c,k] / before.ocean.h[c,k]).tolist()
                rows = []
                for (cell, level), remaining, debit in zip(bad_positions.tolist(), available, cost):
                    x, y, z = before.ocean.mesh.cell_xyz[cell].tolist()
                    edges = (self._active_i == cell) | (self._active_j == cell)
                    pi, pj = self._active_i[edges], self._active_j[edges]
                    near = torch.where(pi == cell, pj, pi)
                    adjacent_e = before.eddy_h[near,level] / before.ocean.h[near,level]
                    rows.append(dict(cell=cell, level=level, lat_deg=math.degrees(math.asin(z)),
                                     lon_deg=math.degrees(math.atan2(y,x)),
                                     depth_m=-float(self._height[cell,level]),
                                     cell_e_J_kg=float(before.eddy_h[cell,level] / before.ocean.h[cell,level]),
                                     neighbor_e_range_J_kg=[float(adjacent_e.min()), float(adjacent_e.max())],
                                     face_kappa_range_m2_s=([float(self._diffusivity[edges,level].min()),
                                                            float(self._diffusivity[edges,level].max())]
                                                            if self._diffusivity is not None else None),
                                     available_J_kg=remaining, debit_J_kg=debit))
                self._failure_rows.append(rows)
                raise

        def apply(self, model, state, dt):
            nonlocal step
            self._failure_rows = []
            end, report = super().apply(model, state, dt)
            step += 1
            psi, horizontal, vertical = self._unlimited
            q = self._length * psi[:,1:-1]
            f = self._final_factor
            # q is the interface (edge length times streamfunction) transport.
            # On the changed stencil also retain the full local denominator.
            row = dict(step=step, dt=float(dt), failures=self._failure_rows)
            row["scaled_interfaces"] = int((f < 1).sum()) if f is not None else 0
            row["eligible_interfaces"] = q.numel()
            row["psi_before_m3"] = float(q.abs().sum()) * dt
            row["psi_cut_m3"] = float((q.abs() * (1-f)).sum()) * dt if f is not None else 0.
            row["psi_before_scaled_m3"] = float(q[f < 1].abs().sum()) * dt if f is not None else 0.
            for field, original, final in (("h", horizontal, report.horizontal_transport),
                                           ("v", vertical, report.vertical_transport)):
                row[field + "_before_m3"] = float(original.abs().sum()) * dt
                row[field + "_change_m3"] = float((final-original).abs().sum()) * dt if f is not None else 0.
            with (output_dir / "gm_transport_audit.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            self._unlimited = self._final_factor = self._length = self._height = self._diffusivity = None
            return end, report

    return AuditedGM


def experiment_sources(parent_sources):
    """Attach the limiter and audit code to the original native source map."""
    if any(parent_sources.get(k) != v for k, v in BOUNDED_ICE_SOURCES.items()):
        raise ValueError("Expected the audited bounded sea-ice phase implementation.")
    parent = {**parent_sources, **DAY11_ICE_SOURCES}
    serialized = json.dumps(parent, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(serialized.encode()).hexdigest() != DONOR_SOURCE_MAP_SHA256:
        raise ValueError("Expected the donor-parcel campaign sources from the day-10 run.")
    return {**parent_sources, EXPERIMENT_SOURCE: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def verify_fork(prior, current):
    """Check the exact full manifests before a one-way ice-phase continuation."""
    native = {name: sha for name, sha in current.items() if name != EXPERIMENT_SOURCE}
    parent = {**native, **DAY11_ICE_SOURCES}
    day11 = {**parent, EXPERIMENT_SOURCE: DAY11_LIMITER_SHA256}
    if current != experiment_sources(native) or prior not in (parent, day11, current):
        raise ValueError("Expected the strict donor parent, day-11 limiter, or identical bounded-ice sources.")
    return {
        "experiment": "donor-parcel GM with local streamfunction funding and bounded sea-ice phase",
        "added_sources": {k: v for k, v in current.items() if k not in prior},
        "modified_sources": {k: {"before": prior[k], "after": v}
                             for k, v in current.items() if k in prior and prior[k] != v},
    }

def run_campaign(argv):
    """Run native restart validation and campaign I/O with the experimental GM."""
    args = argparse.ArgumentParser(add_help=False)
    args.add_argument("--output", type=Path, required=True)
    output = args.parse_known_args(argv)[0].output
    campaign = load_campaign()
    from vfs.ocean.transport import divergence

    limited_gm = make_audited_gm_class(campaign.GMRedi, divergence, output)
    sources = experiment_sources(campaign.source_hashes())
    print("Testing donor-parcel GM with local streamfunction funding.", flush=True)
    with (
        patch.object(campaign, "GMRedi", limited_gm),
        patch.object(campaign, "source_hashes", lambda: sources),
        patch.object(campaign, "verify_compatible_source_delta", verify_fork),
    ):
        return campaign.main(argv)
