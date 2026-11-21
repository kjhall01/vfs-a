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

    def __init__(self, path, measure, summarize, *, substeps=False):
        self.file = open(path, "w", encoding="utf-8")
        self.measure = measure
        self.summarize = summarize
        self.substeps = bool(substeps)
        self.active = False
        self.interval = 0
        self.stage_index = 0
        self.previous = None
        self.interval_start = None
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
        self.external = self.upstream = self.residual = 0.0
        self.active = True
        self._write(dict(
            event="begin", interval=self.interval,
            interval_start_seconds=self.interval_start,
            seconds=int(air.time_seconds),
            energy_J_m2=self.previous["energy"],
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

    def commit(self, ledger, *, seconds):
        """Record an accepted interval and reconcile against campaign totals."""
        if not self.active:
            raise RuntimeError("cannot commit an inactive energy trace")
        total = (float(self.previous["energy"]) - float(ledger["initial"]["energy"])
                 - float(ledger["external"]) - float(ledger["upstream"]))
        self._write(dict(
            event="commit", interval=self.interval,
            interval_start_seconds=self.interval_start,
            seconds=int(seconds),
            stage_count=self.stage_index,
            interval_external_J_m2=self.external,
            interval_upstream_J_m2=self.upstream,
            interval_numerical_residual_J_m2=self.residual,
            campaign_cumulative_numerical_residual_J_m2=total,
        ))
        self.active = False
        return total
