"""Run the real fully coupled benchmark with a deeper ocean–ice RHS clock.

This is observational instrumentation only, layered on profile_coupled.py.
Every wrapped method delegates exactly once with original arguments. No
changes to prognostic equations, limiter, exchange or energy checks.

CLI is IDENTICAL to tools/profile_coupled.py. Run from the repository root:
    python -u -I tools/profile_ocean_rhs.py --config ... --source ... \
        --mesh ... --radiation-data ... --output ...
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import profile_coupled as base


class RHSInstrument(base.Instrument):
    """Observe pure nested regions of the real ocean+ice+snow RHS."""

    def install(self):
        super().install()
        coupled = self.receiver.snow.coupled
        ocean = coupled.ocean
        core = ocean.dynamics

        # snow.tendency delegates to OceanIce.tendency: isolate the latter.
        self._method(coupled, "tendency", "rhs_coupled_ocean_ice")

        # Temperature-coordinate baroclinic/ocean dynamics, with full Gibbs
        # thermal and bounded salt+tracer transport.
        self._method(core, "tendency_with_report", "rhs_ocean_temperature_tendency")
        self._method(core, "fields", "rhs_temperature_entropy_fields")
        from vfs.ocean.stratified import transport
        self._method(transport, "evaluate", "rhs_conservative_transport_total")
        # The Roquet path calls the shared entropy/Gibbs parent evaluator.
        # Its older standalone bounded_transport() helper is not invoked.
        self._method(transport.parent, "evaluate", "rhs_parent_gibbs_transport")
        self._method(transport, "local_target", "rhs_local_energy_target")
        self._method(transport, "_local_target_fields", "rhs_local_target_fields")

        # Native core: transport primitives, Hamiltonian derivatives, physical
        # ocean pressure / balance and limiter work.
        native = ocean.ocean.base
        self._method(native.dynamics, "tendency", "rhs_native_ocean_dynamics")
        if hasattr(native.dynamics, "tendency_terms"):
            self._method(native.dynamics, "tendency_terms", "rhs_native_tendency_terms")
        self._method(ocean.ocean.energy.base, "gradients", "rhs_ocean_energy_gradients")

        # Sea-ice contributions on the same RHS stage; identify graph advection
        # versus ridging/compaction versus energy and drag work.
        self._method(coupled, "ice_transport", "rhs_ice_transport")
        self._method(coupled, "conjugates", "rhs_ice_conjugates")
        self._method(coupled.ridge, "tendency", "rhs_ice_ridging")
        self._method(coupled, "check", "rhs_state_checks")
        self._method(coupled.hamiltonian, "mass", "rhs_ice_mass")
        self._method(coupled.hamiltonian, "metric", "rhs_ice_metric")
        self._method(coupled.hamiltonian, "coefficient", "rhs_ice_coefficients")


if __name__ == "__main__":
    # All output formatting, exact fingerprint checks, staged warmup, true
    # SeaEsfRad calls and strict checkpoint handling remain inherited.
    base.Instrument = RHSInstrument
    raise SystemExit(base.main())
