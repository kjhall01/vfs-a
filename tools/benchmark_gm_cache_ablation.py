"""Compare direct scale selection against halving with the same retry geometry cache.

First check the cached-halving ocean endpoint against ordinary halving over
the whole 1200-second ocean window. Then time paired evolving GM/Redi sequences
with cached halving and direct funding starting at the first strict failure.
"""
from contextlib import nullcontext
from pathlib import Path
import argparse
from statistics import median
import sys
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import benchmark_gm_funding as bench
import gm_direct_funding


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("resume", "manifest", "source", "mesh", "output"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--threads", type=int)
    p.add_argument("--stress-steps", default=12, type=int)
    p.add_argument("--repeats", default=3, type=int)
    p.add_argument("--warmups", default=1, type=int)
    args = p.parse_args()
    if min(args.stress_steps, args.repeats) < 1 or args.warmups < 0:
        p.error("Use positive repeat/step counts and nonnegative warmups.")
    campaign = bench.load_campaign()
    cfg, receiver, sea = bench.read_start(args, campaign)
    well, failed, where = bench.find_inputs(campaign, cfg, receiver, sea)
    results = {
        "strict_failure_step": where, "threads": cfg.threads, "mode": "CPU float64 eager",
        "source_hashes": {f: bench.sha256(TOOLS / f) for f in
                          ("benchmark_gm_funding.py","gm_direct_funding.py","gm_parcel_limiter.py")},
        "cache_ablation_source": bench.sha256(Path(__file__)),
        "stress_steps": args.stress_steps, "pairs": [],
    }
    def trial(method, case, n):
        # Request the built-in halving fallback before any direct scale is
        # computed. The cached geometry and full native checks remain active.
        if method == "cached_halving":
            ctx = patch.object(gm_direct_funding, "_direct_factors", lambda *unused: None)
            base = "direct"
        else:
            ctx = nullcontext()
            base = method
        with ctx:
            metrics, end = bench.trial(campaign, cfg, receiver, sea, (well, failed), case, base,
                                      args.output / case / f"pair-{n}-{method}", args)
        print(f"{case} {n} {method} {metrics['seconds']:.4f} s", flush=True)
        return metrics, end

    with bench.reserve_output(args.output):
        old, end_old = trial("halving", "window", 0)
        cached, end_cached = trial("cached_halving", "window", 0)
        equal = bench.state_equal(end_old, end_cached)
        results["window_comparison"] = dict(halving=old,cached_halving=cached,end_ocean_bitwise_equal=equal)
        if not equal:
            raise AssertionError("Cached halving changed the ocean endpoint.")
        window = args.output / "window"
        original_audit = (window / "pair-0-halving/transport.jsonl").read_bytes()
        cached_audit = (window / "pair-0-cached_halving/transport.jsonl").read_bytes()
        if original_audit != cached_audit:
            raise AssertionError("Cached halving changed a substep transport or budget audit.")
        for n in range(-args.warmups, args.repeats):
            pair = {}
            order = ("cached_halving","direct") if n % 2 == 0 else ("direct","cached_halving")
            for method in order:
                pair[method], _ = trial(method, "sustained", n)
            if n >= 0:
                results["pairs"].append(pair)
                ratios = [p["cached_halving"]["seconds"]/p["direct"]["seconds"] for p in results["pairs"]]
                results["summary"] = dict(paired_speedup_median=median(ratios),
                                          paired_speedup_range=[min(ratios),max(ratios)])
                campaign.atomic_json(args.output/"ablation.json",results)
if __name__ == '__main__':
    with bench.torch.inference_mode():
        main()
