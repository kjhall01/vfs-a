"""Explicit SSP integration for the research coupled ocean–ice dynamics.

This module only combines existing Euler-stage states and wind-work reports.
No tracer limiter, thermodynamics, Hamiltonian, or closure is modified.
"""


def ssprk43_dynamics(model, state, dt, **options):
    """Four-stage third-order SSPRK(4,3), SSP coefficient 2.

    Each forward-Euler stage uses h=dt/2 in *both* the RHS provider and
    model.add(). This matters because ocean tracer admissibility depends on
    the actual carrier/antidiffusive flux over the Euler-stage interval.

      q1   = E_h(q0)
      q2   = E_h(q1)
      q3   = (2/3)q0 + (1/3)E_h(q2)
      q4   = E_h(q3)

    The independent wind-work reports are integrated with the *same*
    Runge–Kutta weights: (1/6, 1/6, 1/6, 1/2). As for SSPRK3, this is a
    third-order approximation to applied boundary work, not exact temporal
    conservation by the explicit integrator.
    """
    h = 0.5 * dt
    r0, p0 = model.tendency(state, h, **options)
    first = model.add(state, r0, h)
    r1, p1 = model.tendency(first, h, **options)
    second = model.add(first, r1, h)
    r2, p2 = model.tendency(second, h, **options)
    middle = model.blend(state, model.add(second, r2, h), 1.0 / 3.0)
    r3, p3 = model.tendency(middle, h, **options)
    end = model.add(middle, r3, h)
    model.check(end)
    wind_work = dt * (
        p0.wind_power / 6.0
        + p1.wind_power / 6.0
        + p2.wind_power / 6.0
        + p3.wind_power / 2.0
    )
    return end, wind_work
