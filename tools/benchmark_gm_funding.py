"""Compare the complete GM/Redi substep on ocean fields from a saved campaign.

Both schemes start each timing pair from the same ocean. Only the reference
advances the input to the next pair. No campaign restart is created.
"""
import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows.coupled import campaign as c
from workflows.coupled.device import execution_context, resolve_device, synchronize


@torch.no_grad()
def benchmark(model, state, config, steps):
    if not config.gm_redi:
        raise ValueError("GM/Redi must be enabled in the saved campaign")
    schemes = ("funded_donor_parcel", "quadratic_funded_donor_parcel")
    operators = [c.gm_redi_operator(replace(config, gm_energy_scheme=s)) for s in schemes]
    interval = config.ocean_dt if config.mesoscale_dt is None else config.mesoscale_dt
    dt = interval / config.eddy_substeps
    device = state.eddy_h.device
    for op in operators:
        op.apply(model, state, dt)
    rows = []
    for step in range(steps):
        results = {}
        order = (0, 1) if step % 2 == 0 else (1, 0)
        for index in order:
            synchronize(device)
            start = perf_counter()
            end, report = operators[index].apply(model, state, dt)
            synchronize(device)
            results[index] = (perf_counter() - start, end, report)
        old_time, reference, old_report = results[0]
        new_time, predicted, new_report = results[1]
        state_equal = all(torch.equal(getattr(reference.ocean, key), getattr(predicted.ocean, key))
                          for key in ("h", "u_e", "temp_h", "salt_h", "tracer_h")
                          if getattr(reference.ocean, key) is not None)
        state_equal &= (torch.equal(reference.tke_h, predicted.tke_h)
                        and torch.equal(reference.eddy_h, predicted.eddy_h))
        different = []
        for field in fields(old_report):
            if field.name == "gm_limiting_passes":
                continue
            a, b = getattr(old_report, field.name), getattr(new_report, field.name)
            if not (torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b):
                different.append(field.name)
        rows.append(dict(original_s=old_time, quadratic_s=new_time,
                         original_retries=old_report.gm_limiting_passes,
                         quadratic_retries=new_report.gm_limiting_passes,
                         state_bitwise_equal=state_equal, different_report_fields=different))
        state = reference
    times = [sum(r[key] for r in rows) for key in ("original_s", "quadratic_s")]
    return dict(dt_seconds=dt, device=str(device), dtype=str(state.eddy_h.dtype),
                threads=torch.get_num_threads(), speedup=times[0] / times[1],
                original_median_s=median(r["original_s"] for r in rows),
                quadratic_median_s=median(r["quadratic_s"] for r in rows), pairs=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("manifest", "checkpoint", "mesh", "source"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        parser.error("steps and threads must be positive")
    manifest = json.loads(args.manifest.read_text())
    payload = c.read_checkpoint(args.checkpoint)
    if payload.get("schema") != 1 or payload["fingerprint"] != manifest["fingerprint"]:
        parser.error("checkpoint and campaign manifest must match")
    config = c.CampaignConfig(**manifest["configuration"])
    config, xyz, geometry = c.load_mesh(config, args.mesh)
    if geometry["sha256"] != manifest["mesh"].get("sha256"):
        parser.error("mesh must match the saved campaign generators")
    device = resolve_device(args.device)
    config = replace(config, device=str(device), compile=False, threads=args.threads)
    with execution_context(device), torch.no_grad():
        _, _, receiver, sea = c.build(config, args.source, None, smoke_no_radiation=True,
                                      generator_xyz=xyz)
        template = sea.ocean
        values = c.copy_tree(payload["ocean"], device=device)
        state = template.with_layers(template.ocean.with_tensors(**values), tke_h=payload["tke_h"].to(device), eddy_h=payload["eddy_h"].to(device))
        result = benchmark(receiver.H.model, state, config, args.steps)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
