"""Explicit event schedule for a single conservative coupled physics interval.

The atmospheric column and its surface Exchange remain one atomic operation.
Independent ocean processes operate on completed ocean-dynamics states.

Each ocean closure cadence divides the coupling interval, so no pending
prognostic tendency, accumulated flux, or partial closure event crosses a
checkpoint. All clocks use integer model seconds, never floating-point modulo.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OceanEvent:
    """Operations at the end of one ocean dynamics timestep."""

    end_seconds: int
    dynamics_dt: int
    mixing_dt: int | None
    mesoscale_dt: int | None


def ocean_events(physics_dt, ocean_dt, mixing_dt, mesoscale_dt):
    """Construct deterministic, ordered ocean events inside a physics interval.

    Each independent closure is applied after dynamics, but only at its own
    cadence. The trailing dt is elapsed *since the prior closure application*.
    This is an operator-split experiment, not a change to closure equations.
    """
    values = (physics_dt, ocean_dt, mixing_dt, mesoscale_dt)
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("ocean event cadences must be positive integer seconds")
    # Normal cadence commensurability is validated on the full configured
    # physics interval. A user-requested stop may leave a shorter terminal
    # interval: flush remaining closure time before committing that stop.
    for name, cadence in (("mixing_dt", mixing_dt), ("mesoscale_dt", mesoscale_dt)):
        if cadence < ocean_dt or cadence % ocean_dt:
            raise ValueError(
                f"{name} must be an integer multiple of ocean_dt"
            )
    since_mix = since_meso = 0
    for elapsed in range(0, physics_dt, ocean_dt):
        step_dt = min(ocean_dt, physics_dt - elapsed)
        end = elapsed + step_dt
        since_mix += step_dt
        since_meso += step_dt
        mix_due = end % mixing_dt == 0 or end == physics_dt
        meso_due = end % mesoscale_dt == 0 or end == physics_dt
        yield OceanEvent(
            end_seconds=end,
            dynamics_dt=step_dt,
            mixing_dt=since_mix if mix_due else None,
            mesoscale_dt=since_meso if meso_due else None,
        )
        if mix_due:
            since_mix = 0
        if meso_due:
            since_meso = 0


def event_due(model_seconds, cadence):
    """Anchor infrequent instantaneous post-exchange operators to model time."""
    if (type(model_seconds) is not int or model_seconds < 0
            or type(cadence) is not int or cadence <= 0):
        raise ValueError("event time and cadence must be nonnegative/positive integers")
    return model_seconds % cadence == 0
