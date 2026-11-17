"""Regional capture, coupling budgets, eager isolation and restart semantics."""
from dataclasses import replace
import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
import yaml

pytest.importorskip('pyam21')
from test_coupled_campaign import source, config
from test_pyam21_workflow import same_tree
from test_radiation_mapping import NativeGridRadiationPackage
from workflows.coupled import campaign as c
from workflows.coupled.compilation import configure_compilation, MESH_KERNELS
from workflows.pyam21.radiation import RadiationSetup, HostRadiationDriver


@pytest.fixture(autouse=True)
def isolated_compiler_cache():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def capture_only(adapter, receiver, *, enabled=False):
    # Full-graph Dynamo capture in ordinary CI; native codegen is exercised by
    # the separately recorded installed-wheel qualification/benchmark driver.
    return configure_compilation(adapter, receiver, enabled=enabled, backend='eager')


def radiating(nml, model, data_dir, **kwargs):
    return RadiationSetup(HostRadiationDriver.from_nml(nml, NativeGridRadiationPackage()),
                          'analytic-native-grid', {'synthetic': True})


def test_opt_in_is_local_idempotent_and_restart_identified(source, config):
    eager = c.build(config, source, None, smoke_no_radiation=True)
    compiled = c.build(config, source, None, smoke_no_radiation=True)
    a, _, r, _ = compiled
    before = c.fingerprint(config, a, {}, True)
    first = configure_compilation(a, r, enabled=True, backend='eager')
    assert first == configure_compilation(a, r, enabled=True, backend='eager')
    assert c.fingerprint(config, a, {}, True) != before
    assert c.fingerprint(replace(config, compile=True), a, {}, True) != before
    for name in MESH_KERNELS:
        assert not hasattr(getattr(eager[0].model.mesh, name), '_torchdynamo_orig_callable')
        assert hasattr(getattr(a.model.mesh, name), '_torchdynamo_orig_callable')
    assert not hasattr(eager[0].model.dynamics.kernel.mixture_gradients, '_torchdynamo_orig_callable')
    for options in ({'enabled': False}, {'enabled': True, 'backend': 'inductor'}):
        with pytest.raises(ValueError, match='rebuild'):
            configure_compilation(a, r, **options)
    with pytest.raises(ValueError, match='boolean'):
        replace(config, compile='yes').validate()


def test_compiled_shared_mesh_retains_adjoint_pairs(source, config):
    a, _, r, _ = c.build(config, source, None, smoke_no_radiation=True)
    configure_compilation(a, r, enabled=True, backend='eager')
    mesh = a.model.mesh
    gen = torch.Generator().manual_seed(37)
    x = torch.randn(mesh.nCells, 5, generator=gen)
    y = torch.randn(mesh.nEdges, 5, generator=gen)
    edge = (mesh.dc_edge * mesh.dv_edge)[:, None]
    left = (edge * mesh.grad(x) * y).sum()
    right = -(mesh.cell_area[:, None] * x * mesh.div(y)).sum()
    torch.testing.assert_close(left, right, rtol=2e-13, atol=1e-6)
    weight = mesh.edge_side_area.sum(-1)[:, None]
    left = (weight * mesh.cell_to_edge(x) * y).sum()
    right = (mesh.cell_area[:, None] * x * mesh.edge_to_cell(y)).sum()
    torch.testing.assert_close(left, right, rtol=2e-13, atol=1e-2)


def test_full_coupling_capture_radiation_cache_and_bitwise_restart(source, config):
    with patch('workflows.pyam21.radiation.build_radiation', radiating):
        a0, air0, r0, sea0 = c.build(config, source, Path('synthetic'))
        with patch.object(c, 'configure_compilation', capture_only):
            a1, air1, r1, sea1 = c.build(replace(config, compile=True), source, Path('synthetic'))
    identity = c.fingerprint(replace(config, compile=True), a1, {}, False)
    with torch.no_grad():
        for _ in range(3):  # fresh optics, cache hit, fresh optics
            air0, sea0, report0 = c.advance(a0, air0, r0, sea0, 60, config)
            saved = c.pack(a1, air1, r1, sea1, identity, {})
            air1, sea1, report1 = c.advance(a1, air1, r1, sea1, 60, config)
            left = c.pack(a0, air0, r0, sea0, 'compare', {})
            right = c.pack(a1, air1, r1, sea1, 'compare', {})
            same_tree(left, right)
            same_tree(report0, report1)
        restored_air, restored_sea, _ = c.unpack(saved, a1, r1, sea1, identity)
        replay_air, replay_sea, replay_report = c.advance(a1, restored_air, r1, restored_sea, 60, config)
        same_tree(report1, replay_report)
        same_tree(c.pack(a1, air1, r1, sea1, identity, {}),
                  c.pack(a1, replay_air, r1, replay_sea, identity, {}))
        with pytest.raises(ValueError, match='does not match'):
            c.unpack(saved, a1, r1, sea1, 'other-mode')
        with pytest.raises(ValueError, match='outside'):
            s = sea1.coupled.ocean.ocean
            r1.H.model.ocean.reference.evaluate(s.salinity(), s.temperature()*0+100)


def test_cli_compile_override_manifest_preflight_and_log(source, config, tmp_path):
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(c.asdict(config)))
    common = ['--source', str(source), '--config', str(path), '--smoke-no-radiation', '--preflight']
    with patch.object(c, 'configure_compilation', capture_only):
        assert c.main(common + ['--output', str(tmp_path/'compiled'), '--compile']) == 0
    directory = tmp_path/'compiled'
    assert json.loads((directory/'configuration.json').read_text())['compile'] is True
    manifest = json.loads((directory/'manifest.json').read_text())['compilation']
    assert manifest['enabled'] and manifest['backend'] == 'eager'
    assert len(manifest['targets']) == 9
    assert 'first kernel calls' in (directory/'run.log').read_text()
    assert json.loads((directory/'status.json').read_text())['restart_replay'] == 'bitwise'
    path.write_text(yaml.safe_dump(c.asdict(replace(config, compile=True))))
    assert c.main(common + ['--output', str(tmp_path/'eager'), '--no-compile']) == 0
    assert json.loads((tmp_path/'eager/manifest.json').read_text())['compilation']['enabled'] is False


def test_compiler_failure_is_not_silently_retried_eager(source, config, tmp_path):
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(c.asdict(config)))
    def broken(function, **options):
        def fail(*args, **kwargs):
            raise RuntimeError('injected compiler failure')
        return fail
    with patch('torch.compile', broken):
        assert c.main(['--source', str(source), '--config', str(path), '--smoke-no-radiation',
                       '--output', str(tmp_path/'failed'), '--preflight', '--compile']) == 1
    status = json.loads((tmp_path/'failed/status.json').read_text())
    assert status['status'] == 'failed' and 'injected compiler failure' in status['error']


@pytest.mark.skipif(os.environ.get('VFS_TEST_INDUCTOR') != '1', reason='opt-in native CPU compiler qualification')
@torch.no_grad()
def test_native_inductor_radiating_intervals_and_restart(source, config):
    with patch('workflows.pyam21.radiation.build_radiation', radiating):
        a0, air0, r0, sea0 = c.build(config, source, Path('synthetic'))
        a1, air1, r1, sea1 = c.build(replace(config, compile=True), source, Path('synthetic'))
    initial0 = c.inventories(a0, air0, r0, sea0)
    initial1 = c.inventories(a1, air1, r1, sea1)
    upstream0 = upstream1 = external0 = external1 = 0.
    for _ in range(3):
        saved = c.pack(a1, air1, r1, sea1, 'native', {})
        air0, sea0, report0 = c.advance(a0, air0, r0, sea0, 60, config)
        air1, sea1, report1 = c.advance(a1, air1, r1, sea1, 60, config)
        external0 += report0['external']; external1 += report1['external']
        upstream0 += report0['upstream']; upstream1 += report1['upstream']
    from validation.coupled_compile.run import tensors
    left = c.pack(a0, air0, r0, sea0, 'compare', {})
    right = c.pack(a1, air1, r1, sea1, 'compare', {})
    for (name, x), (other, y) in zip(tensors(left), tensors(right), strict=True):
        assert name == other
        torch.testing.assert_close(x, y, rtol=3e-11, atol=3e-10, msg=name)
    final0 = c.inventories(a0, air0, r0, sea0)
    final1 = c.inventories(a1, air1, r1, sea1)
    # Compare independently accumulated whole-system inventories/forcing;
    # compiler reassociation is allowed only at floating-point budget scale.
    eps = torch.finfo(torch.float64).eps
    residual0 = final0['energy']-initial0['energy']-external0-upstream0
    residual1 = final1['energy']-initial1['energy']-external1-upstream1
    assert abs(residual1-residual0) <= 512*eps*abs(initial0['energy'])
    for key in ('water', 'salt'):
        assert abs(final1[key]-initial1[key]) <= 512*eps*abs(initial1[key])
    ra, rs, _ = c.unpack(saved, a1, r1, sea1, 'native')
    ra, rs, rr = c.advance(a1, ra, r1, rs, 60, config)
    same_tree(report1, rr)
    same_tree(c.pack(a1,air1,r1,sea1,'compare',{}), c.pack(a1,ra,r1,rs,'compare',{}))
