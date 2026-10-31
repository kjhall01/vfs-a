"""Explicitly configured atmospheric steppers; plain, single-model collaborators.

The public advance(state, dt) interface returns the corresponding dry or moist state. A model binds a
stepper once to its owned dynamics; construct a fresh stepper for another model.
No RK expressions live here. Detailed stage results remain on transport or solver.
"""

from dataclasses import dataclass, field

from .bounded_moist import BoundedGibbs


def _default_ark():
    # Import the experimental solver stack only when explicitly selected.
    from vfs.integrators.tableau import ARK4_TABLEAU

    return ARK4_TABLEAU


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
    """Existing bounded Euler-ledger SSPRK3, with intrinsic heat release."""

    _dynamics: object = field(default=None, init=False, repr=False)
    _implementation: object = field(default=None, init=False, repr=False)

    def _bind(self, dynamics):
        _bind_once(self, dynamics, BoundedGibbs(dynamics))

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
class CenteredIMEX:
    """Existing experimental centered ARK; never a bounded transport option.

    ARK4 is the only retained implicit method. The solver is
    constructed once when the model binds dynamics. Its analytic wave is still
    rebuilt at each outer-step anchor by the unchanged solver implementation.
    stage_validator receives outer dt for rejection, not an Euler ledger budget.
    """

    method: object = field(default_factory=_default_ark)
    solver_config: object = None
    stage_validator: object = None
    _dynamics: object = field(default=None, init=False, repr=False)
    _implementation: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        from vfs.integrators.tableau import ARK4_TABLEAU
        from vfs.atmosphere.imex.solvers import IMEXSolverConfig

        if self.solver_config is not None and not isinstance(
            self.solver_config, IMEXSolverConfig
        ):
            raise TypeError("solver_config must be IMEXSolverConfig or None")
        if self.method is not ARK4_TABLEAU:
            raise ValueError("method must be ARK4_TABLEAU")
        if self.stage_validator is not None and not callable(self.stage_validator):
            raise TypeError("stage_validator must be callable or None")

    _bound_config: object = field(default=None, init=False, repr=False)

    def _bind(self, dynamics):
        from vfs.atmosphere.imex.moist_ark import MoistPEIMEXARK

        if self._dynamics is not None:
            raise ValueError("stepper is already bound; construct a fresh stepper")
        solver = MoistPEIMEXARK(
            dynamics,
            self.solver_config,
            stage_validator=self.stage_validator,
        )
        _bind_once(self, dynamics, solver)
        object.__setattr__(self, "_bound_config", solver.solver_config)

    @property
    def solver(self):
        return self._implementation

    def _check_binding(self, dynamics):
        _check(self, dynamics)
        if self.solver.dynamics is not dynamics:
            raise ValueError("IMEX solver.dynamics no longer matches model dynamics")
        if self.solver.solver_config is not self._bound_config:
            raise ValueError("IMEX solver configuration replaced after binding")
        if (
            self.solver.tableau is not self.method
            or self.solver.stage_validator is not self.stage_validator
        ):
            raise ValueError("IMEX configuration changed after binding")

    def advance(self, state, dt):
        self._check_binding(self._dynamics)
        return self.solver.step(state, dt)


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


@dataclass(frozen=True, eq=False)
class DryIMEX:
    """Conservative dry ARK4 with an explicitly configured analytic wave solve."""
    solver_config: object = None
    _dynamics: object = field(default=None, init=False, repr=False)
    _implementation: object = field(default=None, init=False, repr=False)
    _bound_config: object = field(default=None, init=False, repr=False)

    def _bind(self, dynamics):
        from vfs.atmosphere.imex import DryPEIMEXARK, IMEXSolverConfig
        if self._dynamics is not None:
            raise ValueError("stepper is already bound; construct a fresh stepper")
        config = self.solver_config
        if config is None:
            config = IMEXSolverConfig(preconditioner="hydrostatic_schur")
        if not isinstance(config, IMEXSolverConfig):
            raise TypeError("solver_config must be IMEXSolverConfig or None")
        solver = DryPEIMEXARK(dynamics, config)
        _bind_once(self, dynamics, solver)
        object.__setattr__(self, "_bound_config", config)

    @property
    def solver(self):
        return self._implementation

    def _check_binding(self, dynamics):
        _check(self, dynamics)
        if self.solver.dynamics is not dynamics or self.solver.solver_config is not self._bound_config:
            raise ValueError("dry IMEX collaborator binding changed")
        from vfs.integrators.tableau import ARK4_TABLEAU
        if self.solver.tableau is not ARK4_TABLEAU:
            raise ValueError("dry IMEX requires the retained ARK4 tableau")

    def advance(self, state, dt):
        self._check_binding(self._dynamics)
        return self.solver.step(state, dt)
