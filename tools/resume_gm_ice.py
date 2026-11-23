"""Continue the limited-GM coupled campaign with bounded sea-ice phase changes."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


def find_failed_run(desktop):
    """Find the reported failed interval among Desktop runs and VFS run folders."""
    matches = set()
    for pattern in ("*/status.json", "vfs*/*/status.json", "vfs*/*/*/status.json"):
        for file in desktop.glob(pattern):
            try:
                status = json.loads(file.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(status, dict):
                continue
            restart = file.with_name("last_good_restart.pt")
            if (status.get("status") == "failed" and status.get("last_committed_seconds") == 1028400
                    and "inadmissible coupled stage" in str(status.get("error", ""))
                    and restart.is_file() and file.with_name("manifest.json").is_file()):
                matches.add(restart.resolve())
    if len(matches) != 1:
        candidates = ", ".join(str(p) for p in sorted(matches)) or "none"
        raise ValueError(f"Expected one day-11.903 ice failure on Desktop, found {candidates}. "
                         "Specify --resume /path/to/last_good_restart.pt.")
    return matches.pop()


def main(argv=None):
    desktop = Path.home() / "Desktop"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=Path, help="Checkpoint (default: discover the reported failed run on Desktop)")
    p.add_argument("--manifest", type=Path, help="Defaults to the original segment's manifest.json")
    p.add_argument("--mesh", type=Path, help="Defaults to the original generator file from the manifest")
    p.add_argument("--source", type=Path, default=desktop / "PyAM21")
    p.add_argument("--radiation-data", type=Path, default=desktop / "sea_esf")
    p.add_argument("--output", type=Path, help="New segment directory (default: sibling with -ice-fix suffix)")
    p.add_argument("--stop-after-days", type=float, default=30)
    p.add_argument("--preflight", action="store_true", help="Replay the first interval and check restart equivalence")
    args = p.parse_args(argv)
    try:
        restart = args.resume.expanduser().resolve() if args.resume else find_failed_run(desktop)
    except ValueError as e:
        p.error(str(e))
    run = restart.parent.parent if restart.parent.name == "restarts" else restart.parent
    manifest = (args.manifest or run / "manifest.json").expanduser().resolve()
    if not restart.is_file() or not manifest.is_file():
        p.error(f"Expected a checkpoint at {restart} and a manifest at {manifest}.")
    prior = json.loads(manifest.read_text())
    if args.resume is None:
        import torch
        saved = torch.load(restart, map_location="cpu", weights_only=True, mmap=True)
        if saved["air"]["time_seconds"] != 1028400 or saved["fingerprint"] != prior["fingerprint"]:
            p.error("Discovered checkpoint does not match the failed second and segment manifest.")
        del saved
    output = (args.output or run.with_name(run.name + "-ice-fix")).expanduser().resolve()
    source = args.source.expanduser().resolve()
    tables = args.radiation_data.expanduser().resolve()
    radiation = prior["radiation_enabled"]
    if not source.is_dir():
        p.error(f"Missing PyAM21 directory {source}. Specify the original checkout with --source.")
    if radiation and not tables.is_dir():
        p.error(f"Missing Sea ESF directory {tables}. Specify the original tables with --radiation-data.")
    mesh_path = args.mesh or prior["mesh"].get("path")
    mesh = Path(mesh_path).expanduser().resolve() if mesh_path else None
    if mesh is not None and not mesh.is_file():
        p.error(f"Missing mesh {mesh}. Supply the original generators with --mesh.")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gm_parcel_limiter import run_campaign
    from run_gm_day9 import reserve_output

    print(f"Resuming {restart} through model day {args.stop_after_days:g}. Output: {output}", flush=True)
    with reserve_output(output), TemporaryDirectory() as tmp:
        config = Path(tmp) / "configuration.json"
        config.write_text(json.dumps(prior["configuration"]))
        return run_campaign([
            "--config", str(config), "--source", str(source),
            *(["--radiation-data", str(tables)] if radiation else ["--smoke-no-radiation"]),
            "--resume", str(restart), "--resume-compatible-manifest", str(manifest),
            "--stop-after-days", str(args.stop_after_days), "--output", str(output), "--verbose",
            *(["--mesh", str(mesh)] if mesh else []), *(["--preflight"] if args.preflight else []),
        ])


if __name__ == "__main__":
    raise SystemExit(main())
