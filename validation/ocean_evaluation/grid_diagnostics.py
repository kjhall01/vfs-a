"""Inspect endpoint kinetic-energy concentration and curl length scale.

These are resolution/noise screening measures, not an ocean kinetic spectrum.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from vfs.geometry.mesh import VoronoiMesh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    rows = []
    for path in sorted(a.root.glob("**/last_accepted.npz")):
        info = json.loads(path.with_name("summary.json").read_text())
        if info["status"] != "completed":
            continue
        config = info["config"]
        radius = config["radius"] or (1e6 if config["case"] == "channel" else 6371220.0)
        with np.load(path) as data:
            mesh = VoronoiMesh(
                n_cells=config["cells"],
                radius=radius,
                generator_xyz=data["cell_xyz"],
                lloyd_iters=0,
                rotation_scheme="compatible",
                verbose=False,
            )
            wet = torch.from_numpy(data["wet_cell"])
            active = wet[mesh.cellsOnEdge].all(-1)
            u = torch.from_numpy(data["u_e"])
            h = torch.from_numpy(data["h"])
            he = mesh.cell_to_edge(h)
            hv = he[mesh.edgesOnVertex].mean(1)
            metric = (mesh.dc_edge * mesh.dv_edge)[:, None] * he
            edge_ke = (0.5 * metric * u.square()).sum(-1)[active]
            total = edge_ke.sum()
            n = max(1, int(np.ceil(0.01 * len(edge_ke))))
            length = torch.sqrt(
                (metric * u.square()).sum()
                / (mesh.dual_area[:, None] * hv * mesh.curl(u).square())
                .sum()
                .clamp_min(1e-300)
            )
            active_ids = torch.where(active)[0]
            maxedge = active_ids[u[active].abs().amax(-1).argmax()]
            rows.append(
                dict(
                    run=str(path.parent.relative_to(a.root)),
                    active_edges=int(active.sum()),
                    top_one_percent_edges_KE_fraction=(
                        float(edge_ke.topk(n).values.sum() / total)
                        if total > 0
                        else 0.0
                    ),
                    kinetic_participation_fraction=(
                        float(total.square() / (len(edge_ke) * edge_ke.square().sum()))
                        if total > 0
                        else 0.0
                    ),
                    curl_length_m=float(length),
                    median_edge_distance_m=float(mesh.dc_edge[active].median()),
                    curl_length_grid_ratio=float(
                        length / mesh.dc_edge[active].median()
                    ),
                    max_speed_edge_lon=float(
                        torch.rad2deg(
                            torch.atan2(
                                mesh.edge_xyz[maxedge, 1], mesh.edge_xyz[maxedge, 0]
                            )
                        )
                    ),
                    max_speed_edge_dc_m=float(mesh.dc_edge[maxedge]),
                    max_speed_edge_dv_m=float(mesh.dv_edge[maxedge]),
                    max_speed_edge_lat=float(
                        torch.rad2deg(torch.asin(mesh.edge_xyz[maxedge, 2]))
                    ),
                    min_dual_edge_length_m=float(mesh.dv_edge[active].min()),
                )
            )
    (a.root / "grid_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
