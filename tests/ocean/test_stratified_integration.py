"""Migration equality against the frozen research stack, plus public API checks."""
import importlib.util
from pathlib import Path
import sys
from dataclasses import fields, is_dataclass, replace
import pytest
import torch
from vfs.ocean.stratified import StratifiedOcean, AdaptiveThermalRemapper


@pytest.fixture(scope='module')
def research():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('integration_reference',
        Path(__file__).parents[2]/'tools/probe_ocean_adaptive_thermal.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif is_dataclass(a):
        for f in fields(a):
            if f.name not in ('mesh', 'column'):
                exact(getattr(a, f.name), getattr(b, f.name))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: exact(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): exact(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('kind', ['front', 'near_fresh', 'upper_salt', 'tilted'])
@pytest.mark.parametrize('coast', [False, True])
def test_frozen_step_remap_and_ledgers(research, kind, coast):
    m, x, e = research.p.p.p.p.q.p.case(kind, coast)
    ocean = StratifiedOcean(m.mesh, m.column, domain=m.domain,
                            remapper=AdaptiveThermalRemapper())
    # Copy native contents, avoiding a gratuitous entropy/temperature round trip.
    from vfs.ocean.stratified import EntropyState
    y = EntropyState(x.u_e, x.h, x.entropy_h, x.salt_h, x.tracer_h, x.mesh, x.column)
    exact(ocean.energy.terms(y), e.terms(x))
    for _ in range(2):
        x, ledger = research.p.p.p.p.front.step(m, x, 10., e)
        y, report = ocean.step_with_report(y, 10.)
        exact(x, y); exact(ledger, report.stages)
    target = x.h.sum(-1, keepdim=True)*torch.tensor([.2, .3, .5])
    x, old_report = research.a.AdaptiveThermalRemapper().remap_with_report(m, x, target, e)
    y, new_report = ocean.remap_with_report(y, target)
    exact(x, y); exact(old_report, new_report)


def test_api_rejects_unsupported_inputs(research):
    m, x, e = research.p.p.p.p.q.p.case('front')
    ocean = StratifiedOcean(m.mesh, m.column)
    y = ocean.from_temperature(x.physical_state(e.reference))
    for dt in [0., -1., float('nan'), float('inf')]:
        with pytest.raises(ValueError): ocean.step(y, dt)
    with pytest.raises(TypeError): ocean.step(x, 10.)
    with pytest.raises(ValueError): ocean.step(y, 10., target_h=y.h)
    with pytest.raises(ValueError): ocean.check_state(replace(y, salt_h=y.salt_h*0))
    with pytest.raises(ValueError): ocean.check_state(replace(y, mesh=object()))
    assert ocean.budgets(y)['tracers'] is not None


def test_installed_package_does_not_import_research():
    import subprocess
    result = subprocess.run([sys.executable, '-I', '-c',
        "import sys; from vfs.ocean.stratified import StratifiedOcean; "
        "assert not any('candidate' in k or k.startswith('tools.') for k in sys.modules)"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_composed_step_missing_passive_and_derivative(research):
    m, x, e = research.p.p.p.p.q.p.case('front')
    ocean = StratifiedOcean(m.mesh, m.column, remapper=AdaptiveThermalRemapper())
    physical = x.physical_state(e.reference)
    y = ocean.from_temperature(physical.with_tensors(physical.u_e, physical.h,
        physical.temp_h, physical.salt_h, None))
    direction = y.h*torch.sin(torch.arange(y.h.numel()).reshape_as(y.h))
    alpha = torch.tensor(0., requires_grad=True)
    y = replace(y, entropy_h=y.entropy_h+alpha*direction)
    separate, _ = ocean.step_with_report(y, 10.)
    target = separate.h.sum(-1, keepdim=True)*torch.tensor([.2, .3, .5])
    expected, _ = ocean.remap_with_report(separate, target)
    actual, report = ocean.step_with_report(y, 10., target_h=target)
    exact(expected, actual)
    assert actual.tracer_h is None and report.remap is not None
    derivative = torch.autograd.grad(actual.entropy_h.sum(), alpha)[0]
    assert torch.isfinite(derivative)
