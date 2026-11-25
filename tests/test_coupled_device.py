"""Device ownership, portable checkpoint I/O and opt-in CUDA qualification."""
from dataclasses import replace
from contextlib import nullcontext
import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
import yaml

pytest.importorskip("pyam21")
from test_coupled_campaign import source, config
from test_coupled_compilation import radiating
from workflows.coupled import campaign as c
from workflows.coupled.device import (device_spec, dtype_spec, validate_precision,
    resolve_device, execution_context, runtime_manifest, synchronize)
from workflows.pyam21.state import copy_tree
from workflows.pyam21.radiation import reference_pressures
from validation.coupled_device.run import tensors, equal, differences


@pytest.mark.parametrize("value", ["mps:1", "meta", "cpu:0", "cuda:-1", "nonsense", None])
def test_unsupported_device_is_explicit(value):
    with pytest.raises(ValueError, match="cpu"):
        replace(c.CampaignConfig(), device=value).validate()


def test_device_validation_does_not_require_hardware_and_resolution_never_falls_back():
    assert device_spec("cuda:2") == torch.device("cuda:2")
    assert resolve_device("cpu") == torch.device("cpu")
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
            resolve_device("cuda")
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
        with pytest.raises(ValueError, match="outside"):
            resolve_device("cuda:1")
    assert device_spec("mps") == torch.device("mps")
    with patch("torch.backends.mps.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="MPS requested but unavailable"):
            resolve_device("mps")
    with patch("torch.backends.mps.is_available", return_value=True):
        assert resolve_device("mps") == torch.device("mps:0")


def test_precision_pairs_and_explicit_rejections():
    assert dtype_spec("float32") == dtype_spec(torch.float32) == torch.float32
    assert dtype_spec("float64") == torch.float64
    replace(c.CampaignConfig(), device="mps", dtype="float32").validate()
    with pytest.raises(ValueError, match="MPS does not support float64"):
        replace(c.CampaignConfig(), device="mps").validate()
    for value in ("float16", "bfloat16", "int64", None, torch.float16):
        with pytest.raises(ValueError, match="dtype must"):
            dtype_spec(value)


def test_mps_policy_and_synchronization_without_claiming_hardware_execution(monkeypatch):
    before = torch.are_deterministic_algorithms_enabled()
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    with patch("workflows.coupled.device.resolve_device", return_value=torch.device("mps:0")):
        with execution_context("mps"):
            assert torch.are_deterministic_algorithms_enabled()
            assert runtime_manifest("mps", "float32")["dtype"] == "float32"
        assert torch.are_deterministic_algorithms_enabled() == before
        monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        with pytest.raises(ValueError, match="FALLBACK"):
            with execution_context("mps"):
                pytest.fail("hidden CPU fallback accepted")
    with patch("torch.mps.synchronize") as sync:
        synchronize("mps")
        sync.assert_called_once_with()


def test_cpu_policy_leaves_caller_settings_and_tree_gradients_intact(monkeypatch):
    before = torch.are_deterministic_algorithms_enabled()
    with execution_context("cpu"):
        assert runtime_manifest("cpu")["deterministic_algorithms"] == before
    x = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    original = dict(x=x, nested=[torch.tensor([True]), (torch.tensor([7]), None)])
    copied = copy_tree(original, device="cpu", dtype=torch.float32)
    assert copied["x"].dtype == torch.float32
    assert copied["x"].data_ptr() != x.data_ptr()
    assert copied["nested"][0].dtype == torch.bool
    assert copied["nested"][1][0].dtype == torch.int64
    copied["x"].sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    # Exercise CUDA policy cleanup without pretending to execute a CUDA kernel.
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with patch("workflows.coupled.device.resolve_device", return_value=torch.device("cuda:0")), \
         patch("torch.cuda.device", return_value=nullcontext()):
        with pytest.raises(RuntimeError, match="injected"):
            with execution_context("cuda:0"):
                assert torch.are_deterministic_algorithms_enabled()
                assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
                raise RuntimeError("injected")
        assert torch.are_deterministic_algorithms_enabled() == before
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "unsupported")
        with pytest.raises(ValueError, match="workspace|WORKSPACE"):
            with execution_context("cuda:0"):
                pytest.fail("incompatible workspace accepted")


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@torch.no_grad()
def test_cpu_device_cli_manifest_and_disk_restart(source, config, tmp_path, dtype):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c.asdict(replace(config, device="cuda:0"))))
    output = tmp_path / "cpu"
    assert c.main(["--source", str(source), "--config", str(path), "--device", "cpu",
                   "--dtype", dtype, "--threads", "1", "--output", str(output), "--smoke-no-radiation", "--preflight"]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["configuration"]["device"] == "cpu"
    assert manifest["execution"]["device"] == "cpu"
    assert manifest["configuration"]["dtype"] == manifest["execution"]["dtype"] == dtype
    assert "Execution:" in (output / "run.log").read_text()
    adapter, air, receiver, sea = c.build(config, source, None, smoke_no_radiation=True)
    identity = c.fingerprint(config, adapter, {}, True)
    payload = c.pack(adapter, air, receiver, sea, identity, {})
    c.checkpoint(tmp_path / "restart.pt", payload)
    disk = torch.load(tmp_path / "restart.pt", weights_only=True, map_location="cpu")
    assert equal(payload, disk)
    ra, rs, _ = c.unpack(disk, adapter, receiver, sea, identity)
    assert equal(payload, c.pack(adapter, ra, receiver, rs, identity, {}))


def test_unavailable_cuda_logs_failure_before_build(source, config, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c.asdict(config)))
    with patch("torch.cuda.is_available", return_value=False), patch.object(c, "build") as build:
        assert c.main(["--source", str(source), "--config", str(path), "--device", "cuda:0",
                       "--output", str(tmp_path / "failed"), "--preflight", "--smoke-no-radiation"]) == 1
        build.assert_not_called()
    status = json.loads((tmp_path / "failed/status.json").read_text())
    assert status["last_committed_seconds"] is None
    assert "CUDA requested but unavailable" in status["error"]


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@torch.no_grad()
def test_device_comparison_driver_real_scheduler_with_analytic_optics(source, config, tmp_path, dtype):
    import numpy as np
    from vfs.geometry.mesh.grid import _fibonacci_xyz
    from validation.coupled_device.run import main
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(c.asdict(config)))
    mesh = tmp_path / "mesh.npy"
    np.save(mesh, _fibonacci_xyz(config.cells))
    output = tmp_path / "comparison"
    with patch("workflows.pyam21.radiation.build_radiation", radiating):
        assert main(["--source", str(source), "--radiation-data", "synthetic", "--config", str(path),
                     "--mesh", str(mesh), "--device", "cpu", "--dtype", dtype, "--reference-dtype", "float64",
                     "--steps", "3", "--output", str(output)]) == 0
    report = json.loads((output / "comparison.json").read_text())
    assert report["status"] == "comparison_completed"
    assert report["radiation_enabled"]
    for mode in report["modes"].values():
        assert mode["restart_bitwise"]
        assert len(mode["intervals"]) == 3
        assert mode["steady_median_seconds"] > 0
    if dtype == "float64":
        assert all(value.get("max_absolute", value.get("unequal")) == 0
                   for value in report["state_differences"].values())
    else:
        assert report["modes"]["cpu"]["execution"]["dtype"] == "float64"
        assert report["modes"]["device"]["execution"]["dtype"] == "float32"
        assert any(value.get("max_absolute", 0) > 0 for value in report["state_differences"].values())
    assert (output / "device/final.nc").is_file()


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "mps"])
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_float32_full_physics_radiation_and_restart(source, config, tmp_path, device, compiled):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA hardware")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires MPS hardware")
    compile_env = {"cpu": "VFS_TEST_INDUCTOR", "cuda:0": "VFS_TEST_CUDA_COMPILE", "mps": "VFS_TEST_MPS_COMPILE"}[device]
    if compiled and os.environ.get(compile_env) != "1":
        pytest.skip(f"set {compile_env}=1 for native compiler qualification")
    selected = replace(config, device=device, dtype="float32", compile=compiled)
    default = torch.get_default_dtype()
    with patch("workflows.pyam21.radiation.build_radiation", radiating), execution_context(device):
        adapter, air, receiver, sea = c.build(selected, source, Path("synthetic"))
        assert torch.get_default_dtype() == default
        target = resolve_device(device)
        initial = c.inventories(adapter, air, receiver, sea)
        external = upstream = 0.0
        for _ in range(3):
            c.checkpoint(tmp_path / "input.pt", c.pack(adapter, air, receiver, sea, "fp32", {}))
            air, sea, report = c.advance(adapter, air, receiver, sea, selected.physics_dt, selected)
            external += report["external"]
            upstream += report["upstream"]
            assert report["gmredi_steps"] == selected.eddy_substeps
            assert report["eddy_decay_steps"] == report["ale_steps"] == 1
        payload = c.pack(adapter, air, receiver, sea, "fp32", {})
        for name, value in tensors(payload):
            assert value.device == target, name
            if value.is_floating_point():
                assert value.dtype == torch.float32, name
                assert torch.isfinite(value).all(), name
        final = c.inventories(adapter, air, receiver, sea)
        eps = torch.finfo(torch.float32).eps
        assert abs(final["energy"] - initial["energy"] - external - upstream) <= 512 * eps * abs(initial["energy"])
        for key in ("water", "salt"):
            assert abs(final[key] - initial[key]) <= 512 * eps * abs(initial[key])
        disk = torch.load(tmp_path / "input.pt", map_location="cpu", weights_only=True)
        ra, rs, _ = c.unpack(disk, adapter, receiver, sea, "fp32")
        ra, rs, rr = c.advance(adapter, ra, receiver, rs, selected.physics_dt, selected)
        assert equal(report, rr)
        assert equal(payload, c.pack(adapter, ra, receiver, rs, "fp32", {}))
        c.snapshot(tmp_path / "fp32.nc", adapter, air, receiver, sea)


@torch.no_grad()
def test_float32_budget_policy_still_rejects_unfunded_numerical_increment(source, config):
    from vfs.ocean.mixing.lagged import Factor
    selected = replace(config, dtype="float32")
    adapter, air, receiver, sea = c.build(selected, source, None, smoke_no_radiation=True)
    original = Factor.solve
    def defective(self, rhs):
        return 1.01 * original(self, rhs)
    with patch.object(Factor, "solve", defective):
        with pytest.raises(FloatingPointError, match="lagged local budget failed"):
            c.advance(adapter, air, receiver, sea, selected.physics_dt, selected)


@torch.no_grad()
def test_float32_drag_ke_loss_matches_tke_and_reciprocal_momentum(source, config):
    _, _, receiver, sea = c.build(replace(config, dtype="float32"), source, None, smoke_no_radiation=True)
    coupled = receiver.snow.coupled
    initial = sea.coupled
    active = coupled.ridge.active_edges(initial)
    initial = replace(initial, ice=replace(initial.ice,
        velocity=torch.where(active, 0.05, 0.0).to(initial.ice.velocity)))
    end, report = coupled.drag_step(initial, 3600.0)
    eps = torch.finfo(torch.float32).eps
    assert report.relative_solver_residual <= 32 * eps
    assert report.tke_gain.sum() > 0
    assert report.energy_residual.abs() <= 128 * eps * report.tke_gain.sum()
    Mi = coupled.hamiltonian.metric(initial)
    Mo = coupled.ocean.energy.base.mechanical.metric(initial.ocean.ocean)[:, :1]
    change = Mi * (end.ice.velocity - initial.ice.velocity)
    change += Mo * (end.ocean.ocean.u_e[:, :1] - initial.ocean.ocean.u_e[:, :1])
    assert change.abs().max() <= 32 * eps * (Mi * initial.ice.velocity).abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an allocated CUDA GPU")
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_cuda_full_physics_radiation_cache_disk_restart_and_cpu_comparison(source, config, tmp_path, compiled):
    if compiled and os.environ.get("VFS_TEST_CUDA_COMPILE") != "1":
        pytest.skip("set VFS_TEST_CUDA_COMPILE=1 for native CUDA Inductor qualification")
    # All ocean/ice processes remain enabled. Analytic optics isolate the device
    # boundary; validation/coupled_device/run.py separately requires real tables.
    cuda_config = replace(config, device="cuda:0", compile=compiled)
    before_policy = torch.are_deterministic_algorithms_enabled()
    with patch("workflows.pyam21.radiation.build_radiation", radiating):
        a0, air0, r0, sea0 = c.build(config, source, Path("synthetic"))
        with execution_context("cuda:0"):
            a1, air1, r1, sea1 = c.build(cuda_config, source, Path("synthetic"))
            assert all(x.device.type == "cuda" for _, x in a1.model.named_buffers())
            assert a1.physics.lat.device.type == a1.physics.lon.device.type == "cuda"
            assert a1.mapping.wind.coefficients.device.type == "cuda"
            assert reference_pressures(a1.model.column).shape == (config.atmosphere_levels + 1, 2)
            initial0, initial1 = c.inventories(a0, air0, r0, sea0), c.inventories(a1, air1, r1, sea1)
            external = [0.0, 0.0]
            upstream = [0.0, 0.0]
            for _ in range(3):  # fresh/cache/fresh, with prognostic ice and ALE
                c.checkpoint(tmp_path / "cuda.pt", c.pack(a1, air1, r1, sea1, "cuda", {}))
                air0, sea0, report0 = c.advance(a0, air0, r0, sea0, config.physics_dt, config)
                air1, sea1, report1 = c.advance(a1, air1, r1, sea1, config.physics_dt, cuda_config)
                for i, report in enumerate((report0, report1)):
                    external[i] += report["external"]
                    upstream[i] += report["upstream"]
                    assert report["gmredi_steps"] == config.eddy_substeps
                    assert report["eddy_decay_steps"] == report["ale_steps"] == 1
            payload = c.pack(a1, air1, r1, sea1, "compare", {})
            assert all(x.device == torch.device("cuda:0") for _, x in tensors(payload))
            left = c.pack(a0, air0, r0, sea0, "compare", {})
            right = copy_tree(payload, device="cpu")
            assert differences(left, right)
            for (name, x), (other, y) in zip(tensors(left), tensors(right), strict=True):
                assert name == other
                torch.testing.assert_close(x, y, rtol=3e-11, atol=3e-10, msg=name)
            # Match existing native-compiler comparison gates; do not loosen
            # any solver's own acceptance tolerance to obtain a GPU pass.
            final0, final1 = c.inventories(a0, air0, r0, sea0), c.inventories(a1, air1, r1, sea1)
            eps = torch.finfo(torch.float64).eps
            residual0 = final0["energy"] - initial0["energy"] - external[0] - upstream[0]
            residual1 = final1["energy"] - initial1["energy"] - external[1] - upstream[1]
            assert abs(residual1 - residual0) <= 512 * eps * abs(initial0["energy"])
            for key in ("water", "salt"):
                assert abs(final1[key] - initial1[key]) <= 512 * eps * abs(initial1[key])
            disk = torch.load(tmp_path / "cuda.pt", map_location="cpu", weights_only=True)
            assert all(x.device.type == "cpu" for _, x in tensors(disk))
            ra, rs, _ = c.unpack(disk, a1, r1, sea1, "cuda")
            ra, rs, rr = c.advance(a1, ra, r1, rs, config.physics_dt, cuda_config)
            assert equal(report1, rr)
            assert equal(right, copy_tree(c.pack(a1, ra, r1, rs, "compare", {}), device="cpu"))
            c.snapshot(tmp_path / "cuda.nc", a1, air1, r1, sea1)
    assert torch.are_deterministic_algorithms_enabled() == before_policy
