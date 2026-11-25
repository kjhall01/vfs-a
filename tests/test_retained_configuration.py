"""Small configuration surface; no removed selectors masquerading as support."""
import pytest
from vfs import BoundedSSPRK3
from vfs.atmosphere import BoundedGibbs, MoistConservativeDynamics
@pytest.mark.parametrize('factory', [BoundedSSPRK3, lambda **kw: BoundedGibbs(MoistConservativeDynamics(), **kw)])
@pytest.mark.parametrize('value', [True, False])
def test_heat_is_intrinsic_not_a_configurable_switch(factory, value):
    with pytest.raises(TypeError, match='release_as_heat'):
        factory(release_as_heat=value)
