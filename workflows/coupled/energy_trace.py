"""Opt-in, per-operator energy accounting for coupled aquaplanet diagnostics.

Samples use the campaign's *same total-energy functional* and its per-area
external/upstream convention. A stage residual is observed dH minus identified
external energy and upstream residual, not the error of an individual
continuous-time operator. An SSPRK stage may contain finite-step error.

The trace is append-only JSON Lines, flushed after every stage so that failed
and interrupted intervals retain evidence. Only commit records mean that an
interval was accepted by the driver.
"""

import json
import math


class EnergyTrace:
    """Trace complete coupled energy at sub-operator boundaries.

    measure returns the same inventory mapping used by normal diagnostics;
    summarize returns state extrema and locations. Neither callback may advance
    the model or alter its state.
    """

    def __init__(self, path, measure, summarize, *, substeps=False, ocean_stages=False):
        self.file = open(path, "w", encoding="utf-8")
        self.measure = measure
        self.summarize = summarize
        self.substeps = bool(substeps)
        self.ocean_stages = bool(ocean_stages)
        self.active = False
        self.interval = 0
        self.stage_index = 0
        self.previous = None
        self.interval_start = None
        self.interval_initial_energy = None
        self.external = 0.0
        self.upstream = 0.0
        self.residual = 0.0

    def close(self):
        self.file.close()

    def _write(self, row):
        # Reject NaN/Inf in the numerical ledger: invalid JSON can obscure
        # the first contaminated stage.
        self.file.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        self.file.flush()

    def begin(self, adapter, air, receiver, sea):
        if self.active:
            raise RuntimeError("previous energy-trace interval was not committed")
        self.interval += 1
        self.stage_index = 0
        self.interval_start = int(air.time_seconds)
        self.previous = self.measure(adapter, air, receiver, sea)
        self.interval_initial_energy = float(self.previous["energy"])
        self.external = self.upstream = self.residual = 0.0
        self.active = True
        self._write(dict(
            event="begin", interval=self.interval,
            interval_start_seconds=self.interval_start,
            seconds=int(air.time_seconds),
            energy_J_m2=self.previous["energy"],
            component_energy_J_m2={k: self.previous[k] for k in ("atmosphere_energy", "ocean_energy", "ice_snow_energy") if k in self.previous},
            water_kg_m2=self.previous["water"],
            salt_inventory=self.previous["salt"],
            **self.summarize(adapter, air, receiver, sea),
        ))

    def observe(self, stage, adapter, air, receiver, sea, *,
                external=0.0, upstream=0.0, details=None):
        if not self.active:
            raise RuntimeError("energy trace requires begin()")
        current = self.measure(adapter, air, receiver, sea)
        before = float(self.previous["energy"])
        after = float(current["energy"])
        external = float(external)
        upstream = float(upstream)
        values = (before, after, external, upstream)
        if not all(math.isfinite(v) for v in values):
            raise FloatingPointError(f"nonfinite energy ledger at {stage}: {values}")
        change = after - before
        residual = change - external - upstream
        self.external += external
        self.upstream += upstream
        self.residual += residual
        self.stage_index += 1
        self._write(dict(
            event="stage", stage=str(stage), interval=self.interval,
            stage_index=self.stage_index,
            interval_start_seconds=self.interval_start,
            seconds=int(air.time_seconds),
            energy_before_J_m2=before,
            energy_after_J_m2=after,
            component_delta_J_m2={k: current[k] - self.previous[k] for k in ("atmosphere_energy", "ocean_energy", "ice_snow_energy") if k in current and k in self.previous},
            delta_energy_J_m2=change,
            external_J_m2=external,
            upstream_J_m2=upstream,
            stage_numerical_residual_J_m2=residual,
            interval_residual_J_m2=self.residual,
            interval_external_J_m2=self.external,
            interval_upstream_J_m2=self.upstream,
            delta_water_kg_m2=float(current["water"] - self.previous["water"]),
            delta_salt_inventory=float(current["salt"] - self.previous["salt"]),
            details=details or {},
            **self.summarize(adapter, air, receiver, sea),
        ))
        self.previous = current
        return residual

    def commit(self, ledger, *, seconds, report=None):
        """Record an accepted interval and reconcile against campaign totals."""
        if not self.active:
            raise RuntimeError("cannot commit an inactive energy trace")
        total = (float(self.previous["energy"]) - float(ledger["initial"]["energy"])
                 - float(ledger["external"]) - float(ledger["upstream"]))
        expected_external = self.external if report is None else float(report["external"])
        expected_upstream = self.upstream if report is None else float(report["upstream"])
        independently_computed = (
            float(self.previous["energy"]) - self.interval_initial_energy
            - expected_external - expected_upstream
        )
        self._write(dict(
            event="commit", interval=self.interval,
            interval_start_seconds=self.interval_start,
            seconds=int(seconds),
            stage_count=self.stage_index,
            interval_external_J_m2=self.external,
            interval_upstream_J_m2=self.upstream,
            interval_numerical_residual_J_m2=self.residual,
            report_external_minus_traced_J_m2=expected_external - self.external,
            report_upstream_minus_traced_J_m2=expected_upstream - self.upstream,
            closure_reconciliation_J_m2=self.residual - independently_computed,
            campaign_cumulative_numerical_residual_J_m2=total,
        ))
        self.active = False
        return total
