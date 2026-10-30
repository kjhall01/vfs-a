"""Small configuration surface; no removed selectors masquerading as support."""
from dataclasses import fields
import pytest
from vfs import DryAtmosphere, MoistAtmosphere, DryIMEX, CenteredIMEX, BoundedSSPRK3
from vfs.atmosphere import BoundedGibbs, MoistConservativeDynamics
from vfs.imex import IMEXSolverConfig, DryPEIMEXARK, MoistPEIMEXARK
from test_conservative_imex import setup
from test_moist_imex import setup as moist_setup


@pytest.mark.parametrize('keyword', ['implicit_operator', 'compile_kernels', 'compile_backend', 'compile_mode'])
def test_removed_imex_options_are_rejected(keyword):
    assert keyword not in {f.name for f in fields(IMEXSolverConfig)}
    with pytest.raises(TypeError, match=keyword):
        IMEXSolverConfig(**{keyword: None})


def test_valid_low_level_defaults_and_unchanged_public_settings():
    dynamics, dry, _ = setup()
    moist_dynamics, moist, _ = moist_setup()
    for solver, state in ((DryPEIMEXARK(dynamics), dry), (MoistPEIMEXARK(moist_dynamics, IMEXSolverConfig()), moist)):
        _, info = solver.step_with_info(state, 1.)
        assert info.converged
    dry_model = DryAtmosphere(dry.mesh, dry.column, stepper=DryIMEX())
    moist_model = MoistAtmosphere(moist.mesh, moist.column, stepper=CenteredIMEX())
    d = dry_model.stepper.solver.solver_config
    m = moist_model.stepper.solver.solver_config
    assert (d.rtol, d.atol, d.max_iterations, d.restart, d.preconditioner) == (None, 0., 40, 20, 'hydrostatic_schur')
    assert (m.rtol, m.atol, m.max_iterations, m.restart, m.preconditioner) == (1e-9, 0., 80, 30, 'hydrostatic_schur')


@pytest.mark.parametrize('factory', [BoundedSSPRK3, lambda **kw: BoundedGibbs(MoistConservativeDynamics(), **kw)])
@pytest.mark.parametrize('value', [True, False])
def test_heat_is_intrinsic_not_a_configurable_switch(factory, value):
    with pytest.raises(TypeError, match='release_as_heat'):
        factory(release_as_heat=value)
