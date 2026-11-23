"""Observe rejection and recovery without changing the configured ocean operator."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("pyam21")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import diagnose_gm_budget as diag


def test_report_only_an_unrecovered_native_budget_failure(tmp_path):
    class Probe:
        # Use the real finite-work settlement for a deterministic one-cell case.
        _settle = staticmethod(diag.campaign.GMRedi._settle)

        def apply(self, model, state, recover):
            try:
                self._settle(model, state, state, torch.zeros_like(state.eddy_h), "GM")
            except FloatingPointError as error:
                if recover:
                    return state
                raise FloatingPointError("funding retries exhausted") from error

    h = torch.ones((1, 2), dtype=torch.float64)
    oce = SimpleNamespace(h=h, mesh=SimpleNamespace(cell_xyz=torch.tensor([[1., 0., 0.]])))
    state = SimpleNamespace(ocean=oce, eddy_h=torch.tensor([[1., 0.25]], dtype=h.dtype))
    H = SimpleNamespace(mass_area=lambda s: torch.ones_like(s.h[:, :1]),
                        finite_change=lambda pre, post: torch.tensor([[0., 0.5]], dtype=h.dtype))
    model = SimpleNamespace(hamiltonian=H)
    dst = tmp_path / "gm_budget_failure.json"
    operator = diag.reporting_operator(Probe, dst)()
    assert operator.apply(model, state, True) is state
    assert not dst.exists()
    with pytest.raises(FloatingPointError, match="retries exhausted"):
        operator.apply(model, state, False)
    report = json.loads(dst.read_text())
    assert report["stage"] == "GM" and report["gm_redi_apply_number"] == 2
    assert report["negative_cell_layers"] == 1
    row, = report["worst_cell_layers"]
    assert (row["cell"], row["layer"], row["deficit_J_kg"]) == (0, 1, 0.25)


def test_cli_propagates_exit_code_and_restores_operator(monkeypatch, tmp_path):
    native = diag.campaign.GMRedi
    args = ["--output", str(tmp_path)]
    def fail(argv):
        assert argv == args and issubclass(diag.campaign.GMRedi, native)
        assert diag.campaign.GMRedi is not native
        return 1

    monkeypatch.setattr(diag.campaign, "main", fail)
    assert diag.main(args) == 1
    assert diag.campaign.GMRedi is native
