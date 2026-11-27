"""Explicitly configured atmospheric steppers; plain, single-model collaborators.

The public advance(state, dt) interface returns the corresponding dry or moist state. A model binds a
stepper once to its owned dynamics; construct a fresh stepper for another model.
No RK expressions live here. Detailed stage results remain on transport or solver.
"""

from dataclasses import dataclass, field

from .bounded_moist import BoundedGibbs


def _bind_once(stepper, dynamics, implementation):
    if stepper._dynamics is not None:
        raise ValueError("stepper is already bound; construct a fresh stepper")
    object.__setattr__(stepper, "_dynamics", dynamics)
    object.__setattr__(stepper, "_implementation", implementation)


def _check(stepper, dynamics):
    if stepper._dynamics is None or stepper._dynamics is not dynamics:
        raise ValueError("stepper must remain bound to this model's dynamics")


@dataclass(frozen=True, eq=False)
class BoundedSSPRK3:
    """Bounded Euler-ledger SSPRK3, optionally donor-cell at the top wind interface."""

    top_momentum_upwind: bool = field(default=False, kw_only=True)
    _dynamics: object = field(default=None, init=False, repr=False)
    _implementation: object = field(default=None, init=False, repr=False)

    def _bind(self, dynamics):
        _bind_once(self, dynamics, BoundedGibbs(dynamics, top_momentum_upwind=self.top_momentum_upwind))

    @property
    def transport(self):
        return self._implementation

    def _check_binding(self, dynamics):
        _check(self, dynamics)
        if self.transport.base is not dynamics:
            raise ValueError("bounded transport.base no longer matches model dynamics")

    def advance(self, state, dt):
        self._check_binding(self._dynamics)
        return self.transport.step(state, dt)


@dataclass(frozen=True, eq=False)
class DrySSPRK3:
    """The retained conservative dry RHS with the existing SSPRK3 formula."""
    _dynamics: object = field(default=None, init=False, repr=False)
    _implementation: object = field(default=None, init=False, repr=False)

    def _bind(self, dynamics):
        _bind_once(self, dynamics, dynamics)

    def _check_binding(self, dynamics):
        _check(self, dynamics)
        if self._implementation is not dynamics:
            raise ValueError("dry stepper no longer matches model dynamics")

    def advance(self, state, dt):
        from vfs.integrators.explicit import ssprk3
        self._check_binding(self._dynamics)
        return ssprk3(self._dynamics, state, dt)
