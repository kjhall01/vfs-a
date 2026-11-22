"""Resume the day-10 parcel-GM checkpoint with local flux limiting through day 11."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


def main(argv=None):
    desktop = Path.home() / "Desktop"
    runs = desktop / "vfs-a"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", type=Path, default=runs / "gm-parcel-day30/last_good_restart.pt",
                        help="Checkpoint from the failed strict parcel-GM segment")
    parser.add_argument("--manifest", type=Path, help="Manifest from that segment (defaults to manifest.json beside the checkpoint)")
    parser.add_argument("--mesh", type=Path, help="Saved generators (defaults to the path in the manifest)")
    parser.add_argument("--source", type=Path, default=desktop / "PyAM21", help="Original PyAM21 source checkout")
    parser.add_argument("--radiation-data", type=Path, default=desktop / "sea_esf", help="Original Sea ESF tables")
    parser.add_argument("--output", type=Path, default=runs / "gm-parcel-limiter-day11", help="New segment directory")
    args = parser.parse_args(argv)

    restart = args.resume.expanduser().resolve()
    manifest = (args.manifest or restart.with_name("manifest.json")).expanduser().resolve()
    source = args.source.expanduser().resolve()
    tables = args.radiation_data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path, option in ((restart, "--resume"), (manifest, "--manifest")):
        if not path.is_file():
            parser.error(f"Missing {path}. Specify its location with {option}.")
    prior = json.loads(manifest.read_text())
    mesh = (args.mesh or Path(prior["mesh"]["path"])).expanduser().resolve()
    if not mesh.is_file():
        parser.error(f"Missing {mesh}. Specify the generators with --mesh.")
    for path, option in ((source, "--source"), (tables, "--radiation-data")):
        if not path.is_dir():
            parser.error(f"Missing {path}. Specify its directory with {option}.")

    import torch
    payload = torch.load(restart, weights_only=True, map_location="cpu")
    if (payload["air"]["time_seconds"] != 936000 or payload["ledger"]["steps"] != 780
            or payload["fingerprint"] != prior["fingerprint"]):
        parser.error("Expected the step-780 checkpoint at second 936000 and its original manifest.")
    del payload

    # Isolated Python omits both the checkout and this script's directory.
    tools = Path(__file__).resolve().parent
    sys.path[:0] = [str(tools), str(tools.parent)]
    from gm_parcel_limiter import run_campaign
    from run_gm_day9 import reserve_output

    print(f"Starting the parcel-GM local limiter at second 936000 from {restart}", flush=True)
    print(f"Day-11 output: {output}", flush=True)
    with reserve_output(output), TemporaryDirectory() as tmp:
        config = Path(tmp) / "configuration.json"
        config.write_text(json.dumps(prior["configuration"]))
        return run_campaign([
            "--config", str(config), "--mesh", str(mesh), "--source", str(source),
            "--radiation-data", str(tables), "--resume", str(restart),
            "--resume-compatible-manifest", str(manifest), "--stop-after-days", "11",
            "--output", str(output), "--verbose",
        ])


if __name__ == "__main__":
    raise SystemExit(main())
