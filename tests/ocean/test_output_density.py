"""Output density follows the selected equations, including its gradients."""
import pytest
import torch

from vfs.geometry.remap import LatLonGrid, MeshToLatLonKNN
from vfs.ocean import Ocean, OceanColumn, OceanState
from vfs.ocean.conservative import ConstantDensityDynamics
from vfs.ocean.dynamics import MultilayerOceanDynamics
from vfs.ocean.regrid import state_to_latlon
from vfs.geometry.mesh import VoronoiMesh


@pytest.mark.parametrize('equations', ['constant', 'linear', 'jmd95'])
def test_model_output_uses_selected_density_and_preserves_gradients(equations):
    torch.set_default_dtype(torch.float64)
    mesh = VoronoiMesh(n_cells=16, lloyd_iters=0, verbose=False)
    column = OceanColumn(2, total_depth=2000.)
    dynamics = (ConstantDensityDynamics() if equations == 'constant' else
                MultilayerOceanDynamics(eos=equations, alpha=3e-4))
    model = Ocean(mesh, column, dynamics=dynamics)
    h = torch.full((mesh.nCells, 2), 1000.)
    temperature = (10. + 4. * mesh.cell_xyz[:, :1]).expand_as(h).clone().requires_grad_()
    salinity = (35. + mesh.cell_xyz[:, 1:2]).expand_as(h)
    state = model.state(u_e=torch.zeros(mesh.nEdges, 2), h=h,
                        temperature=temperature, salinity=salinity)
    remapper = MeshToLatLonKNN.from_mesh(mesh, LatLonGrid.regular(6, 12))
    output = model.to_latlon(state, remapper, fields='all')
    expected = remapper(model.physical_fields(state)['density'])
    torch.testing.assert_close(output['density'].values, expected, rtol=0, atol=0)
    assert output['density'].units == 'kg m-3'
    assert output['temperature'].units == 'degC'
    assert output['attrs']['density_convention'] == 'selected_dynamics'
    assert output['attrs']['conservative'] is False
    legacy = state_to_latlon(remapper, state, fields='density')
    assert legacy['attrs']['density_convention'] == 'legacy_state_linear_eos'
    assert not torch.allclose(legacy['density'].values, expected, rtol=1e-12, atol=0)
    if equations == 'constant':
        torch.testing.assert_close(expected, torch.full_like(expected, dynamics.rho0))
        assert not expected.requires_grad
    else:
        actual_grad, = torch.autograd.grad(output['density'].values.square().sum(), temperature,
                                           retain_graph=True)
        expected_grad, = torch.autograd.grad(expected.square().sum(), temperature)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        assert torch.isfinite(actual_grad).all() and torch.count_nonzero(actual_grad)
    wrong = OceanState(state.u_e, state.h, state.temp_h, state.salt_h,
                       mesh=mesh, column=OceanColumn(2, total_depth=2000.))
    with pytest.raises(ValueError, match='exact mesh and column'):
        model.to_latlon(wrong, remapper, fields='density')


def test_density_provider_is_not_called_for_other_fields():
    from test_conservative import case
    model, state = case()
    remapper = MeshToLatLonKNN.from_mesh(state.mesh, LatLonGrid.regular(6, 12))
    def unexpected(_):
        raise AssertionError('density was not requested')
    output = state_to_latlon(remapper, state, fields='temperature', density_provider=unexpected)
    assert 'density' not in output


def test_output_capture_allows_only_the_documented_metadata_addition():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / 'tools/capture_io_ownership.py'
    spec = importlib.util.spec_from_file_location('output_capture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = {'conservative': False}
    new = {**old, 'density_convention': 'legacy_state_linear_eos'}
    location = 'output/ocean_output_contract/attrs'
    assert module.compare(new, old, location) == 0
    with pytest.raises(AssertionError):
        module.compare({**new, 'density_convention': 'wrong'}, old, location)
    with pytest.raises(AssertionError):
        module.compare({**new, 'extra': True}, old, location)
    with pytest.raises(AssertionError):
        module.compare(new, old, 'output/atmosphere/attrs')
