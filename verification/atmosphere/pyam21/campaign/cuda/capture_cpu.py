"""Capture unchanged CPU behavior against the pre-CUDA workflow checkout.

Uses analytic radiation through the real scheduler: fresh/cache/cache/fresh at
the campaign's 1200/3600-second cadence. Supply a checkout or copied workflow
tree as --root; run with an installed VFS wheel outside the source checkout.
"""
import argparse
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
args = parser.parse_args()
sys.path.extend([str(args.root), str(args.root / "tests")])
from workflows.coupled import campaign as c
from test_coupled_campaign import source
from test_coupled_compilation import radiating

with tempfile.TemporaryDirectory() as directory, torch.no_grad():
    upstream = source.__wrapped__(Path(directory))
    config = c.CampaignConfig(cells=32, lloyd_iterations=0)
    if args.dtype != "float64":
        from dataclasses import replace
        config = replace(config, dtype=args.dtype)
    with patch("workflows.pyam21.radiation.build_radiation", radiating):
        a, air, r, sea = c.build(config, upstream, Path("synthetic"))
    records = [dict(state=c.pack(a, air, r, sea, "comparison", {}), inventories=c.inventories(a, air, r, sea))]
    for _ in range(4):
        air, sea, report = c.advance(a, air, r, sea, config.physics_dt, config)
        records.append(dict(state=c.pack(a, air, r, sea, "comparison", {}),
                            inventories=c.inventories(a, air, r, sea), report=report))
    torch.save(records, args.output)
print(f"Captured {len(records)} states on CPU; all configured processes enabled")
