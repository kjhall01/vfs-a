"""Render saved native-cell surface elevation; no spatial interpolation."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    info = json.loads((a.run / "summary.json").read_text())
    config = info["config"]
    if config["case"] != "wave":
        p.error("select a wave run")
    label = info["variant"] + (" + sink" if config["biharmonic_viscosity"] else "")
    paths = [a.run / "initial.npz", *sorted(a.run.glob("state-*s.npz"))]
    with np.load(paths[0]) as d:
        xyz = d["cell_xyz"]
        limit = float(abs(d["eta"]).max())
    lon = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    lat = np.rad2deg(np.arcsin(xyz[:, 2]))
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    dots = ax.scatter(
        lon, lat, c=np.zeros_like(lat), s=70, cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude", ylabel="Latitude")
    ax.grid(alpha=0.15)
    fig.colorbar(dots, ax=ax, label="Surface elevation (m)")
    frames = []
    for path in paths:
        seconds = 0 if path.name == "initial.npz" else int(path.stem.split("-")[1][:-1])
        with np.load(path) as d:
            dots.set_array(d["eta"].reshape(-1))
        ax.set_title(
            f"Rotating wave, {label} — day {seconds/86400:.2f}\n"
            f"{config['cells']} native cell centers; {config['layers']} layers; dt={config['dt']:g} s"
        )
        fig.canvas.draw()
        frames.append(
            Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert("RGB")
        )
    frames[0].save(
        a.output, save_all=True, append_images=frames[1:], duration=200, loop=0
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
