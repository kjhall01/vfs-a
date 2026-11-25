"""Named NetCDF fields shared by instantaneous output and lossless coupled restarts."""
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from .grid_io import atomic_path, horizontal_coords, write_netcdf


def array(tensor):
    return tensor.detach().cpu().numpy()


def grid_coordinates(adapter, air, receiver, sea):
    """Static mesh and column coordinates, independent of output time."""
    mesh = adapter.model.mesh
    air_column = adapter.model.column
    ocean = sea.coupled.ocean.ocean
    column = ocean.column
    coords = horizontal_coords(array(mesh.cell_xyz))
    edge_xyz = array(mesh.edge_xyz)
    edge_coords = horizontal_coords(edge_xyz)
    vertex = horizontal_coords(array(mesh.vertex_xyz))
    wet = np.ones(mesh.nCells, dtype=bool) if sea.land is None else array(receiver.land_surface.wet)
    depth = array(column.total_depth_like(ocean.h))[:, 0] * wet
    nominal = array(adapter.campaign_ocean_thickness)
    reference = (array(column.layer_reference) * wet[:, None] if column.layer_reference is not None
                 else depth[:, None] * (nominal / nominal.sum()))
    coords.update(
        edge=np.arange(mesh.nEdges),
        edge_xyz=(("edge", "xyz"), edge_xyz, {"units": "1"}),
        edge_latitude=("edge", edge_coords["latitude"][1], edge_coords["latitude"][2]),
        edge_longitude=("edge", edge_coords["longitude"][1], edge_coords["longitude"][2]),
        cell_area=("cell", array(mesh.cell_area), {"units": "m2"}),
        cells_on_edge=(("edge", "pair"), array(mesh.cellsOnEdge), {"start_index": 0}),
        vertex_latitude=("vertex", vertex["latitude"][1], vertex["latitude"][2]),
        vertex_longitude=("vertex", vertex["longitude"][1], vertex["longitude"][2]),
        vertices_on_edge=(("edge", "pair"), array(mesh.verticesOnEdge),
                          {"cf_role": "edge_node_connectivity", "start_index": 0}),
        vertices_on_cell=(("cell", "max_vertices"), array(mesh.verticesOnCell),
                          {"cf_role": "face_node_connectivity", "start_index": 0}),
        vertices_per_cell=("cell", array(mesh.nEdgesOnCell)),
        mesh=((), np.int32(0), {"cf_role": "mesh_topology", "topology_dimension": 2,
                                "face_dimension": "cell", "edge_dimension": "edge",
                                "node_coordinates": "vertex_longitude vertex_latitude",
                                "face_coordinates": "longitude latitude",
                                "edge_coordinates": "edge_longitude edge_latitude",
                                "face_node_connectivity": "vertices_on_cell",
                                "edge_node_connectivity": "vertices_on_edge"}),
        air_level=np.arange(air.atmosphere.u_e.shape[1]),
        air_interface=np.arange(air_column.A_interface.numel()),
        air_hybrid_a=("air_interface", array(air_column.A_interface), {"units": "Pa"}),
        air_hybrid_b=("air_interface", array(air_column.B_interface), {"units": "1"}),
        ocean_level=np.arange(column.n_layers),
        ocean_nominal_thickness=("ocean_level", nominal, {"units": "m"}),
        ocean_reference_thickness=(("cell", "ocean_level"), reference, {"units": "m"}),
        ocean_depth_m=("cell", depth, {"units": "m", "positive": "down"}),
        ocean_active_layer_count=("cell", np.count_nonzero(reference, axis=1).astype("i4")),
        land_mask=("cell", (~wet).astype("i1"), {"flag_values": np.array([0, 1], dtype="i1"), "flag_meanings": "ocean land"}),
        surface_height_m=("cell", np.broadcast_to(array(air_column.phi_s).reshape(-1), (mesh.nCells,))
                          / adapter.model.dynamics.kernel.gravity, {"units": "m", "positive": "up"}),
        category=("category", ["thin", "thick"]),
    )
    if sea.land is not None:
        dz = np.asarray(receiver.land_surface.column.thickness)
        coords.update(land_cell_index=("land_cell", np.flatnonzero(~wet), {"start_index": 0}),
                      soil_layer=("soil_layer", dz.cumsum() - dz / 2, {"units": "m", "positive": "down"}),
                      soil_layer_thickness=("soil_layer", dz, {"units": "m"}))
    ds = xr.Dataset(coords=coords, attrs={
        "grid": "spherical Voronoi C-grid", "sphere_radius_m": mesh.radius,
        "rotation_scheme": mesh.rotation_scheme, "calendar": "julian",
        "atmosphere_vertical_coordinate": "hybrid pressure at interfaces: p = air_hybrid_a + air_hybrid_b * surface_pressure",
        "ocean_vertical_coordinate": "partial cells" if column.layer_reference is not None else "stretched layers",
    })
    ds.vertices_on_cell.encoding["_FillValue"] = -1
    if getattr(adapter, "campaign_mesh_sha256", None):
        ds.attrs["mesh_sha256"] = adapter.campaign_mesh_sha256
    if hasattr(adapter, "geography_hash"):
        ds.attrs["geography_sha256"] = adapter.geography_hash
    return ds


def clock_attrs(adapter):
    y, m, d, h, minute, sec = adapter.configuration["start"]
    return {"units": f"seconds since {y:04d}-{m:02d}-{d:02d} {h:02d}:{minute:02d}:{sec:02d}", "calendar": "julian"}


# Dimensions follow the native staggering. Keep the singleton axes in the stored state.
FIELDS = {
    "air_atmosphere_u_e": (("edge", "air_level"), "m s-1"),
    "air_atmosphere_ps": (("cell", "singleton"), "Pa"),
    "air_atmosphere_entropy_content": (("cell", "air_interface"), "Pa J kg-1 K-1"),
    "air_atmosphere_water_content": (("cell", "air_interface", "water_species"), "Pa"),
    "air_cloud_fraction": (("cell", "air_interface"), "1"),
    "ocean_u_e": (("edge", "ocean_level"), "m s-1"),
    "ocean_h": (("cell", "ocean_level"), "m"),
    "ocean_temp_h": (("cell", "ocean_level"), "m degC"),
    "ocean_salt_h": (("cell", "ocean_level"), "m g kg-1"),
    "ocean_tracer_h": (("cell", "ocean_level", "ocean_tracer"), None),
    "eddy_h": (("cell", "ocean_level"), "m3 s-2"),
    "tke_h": (("cell", "ocean_level"), "m3 s-2"),
    "ice_area": (("cell", "category"), "1"),
    "ice_volume": (("cell", "category"), "m"),
    "ice_velocity": (("edge", "ice_level"), "m s-1"),
    "snow_mass": (("cell", "category"), "kg m-2"),
    "cold_content": (("cell", "category"), "J m-2"),
    "land_skin": (("land_cell",), "K"), "land_snow": (("land_cell",), "kg m-2"),
    "land_canopy": (("land_cell",), "kg m-2"),
    "land_soil_water": (("land_cell", "soil_layer"), "kg m-2"),
    "land_soil_temperature": (("land_cell", "soil_layer"), "K"),
}


def _field_layout(name, value):
    if name in FIELDS:
        return FIELDS[name]
    if name.startswith(("air_memory_", "air_surface_", "air_radiation_")):
        if value.ndim == 1:
            return ("cell",), None
        if value.ndim == 2 and name.startswith("air_memory_"):
            return ("cell", "air_level"), None
        if value.ndim == 2 and name.startswith("air_radiation_saved_"):
            key = name.removeprefix("air_radiation_saved_")
            if key in {"tdtsw", "tdtlw", "tdt_rad", "sw_heating_save", "tot_heating_save"}:
                return ("cell", "radiation_level"), "K s-1"
            if key == "vfs_radiation_pflux":
                return ("cell", "radiation_interface"), "Pa"
            if key in {"ufsw", "dfsw", "flxnet", "ufsw_save", "dfsw_save"}:
                return ("cell", "radiation_interface"), "W m-2"
    # Future component caches can retain axes with no known physical interpretation.
    return tuple(f"{name}_axis_{i}" for i in range(value.ndim)), None


def attach_mesh_attributes(ds):
    """Associate native fields with the faces and edges of the topology variable."""
    if "mesh" in ds.coords:
        for field in ds.data_vars.values():
            if "cell" in field.dims:
                field.attrs.update(mesh="mesh", location="face")
            elif "edge" in field.dims:
                field.attrs.update(mesh="mesh", location="edge")


def checkpoint(path, payload, *, grid=None, time_attrs=None):
    """Write a complete restart. Explicit .pt paths retain the legacy format."""
    path = Path(path)
    if path.suffix == ".pt":
        from workflows.pyam21.state import copy_tree
        with atomic_path(path) as tmp:
            torch.save(copy_tree(payload, device="cpu"), tmp)
        return
    if path.suffix != ".nc":
        raise ValueError("expected a .nc or .pt restart")
    ds = xr.Dataset() if grid is None else grid.copy(deep=False)
    ds = ds.assign_coords(water_species=["vapor", "liquid", "ice"],
                          time=xr.Variable((), payload["air"]["time_seconds"], attrs=time_attrs))
    ds.attrs.update(vfs_format="coupled_restart_v1", fingerprint=payload["fingerprint"])

    def encode(obj, keys):
        if isinstance(obj, dict):
            return ["dict", {k: encode(v, (*keys, k)) for k, v in obj.items()}]
        if isinstance(obj, (tuple, list)):
            return [type(obj).__name__, [encode(v, (*keys, str(i))) for i, v in enumerate(obj)]]
        if isinstance(obj, str) or obj is None:
            return ["literal", obj]
        name = "_".join(keys)
        if name in ds.variables:
            raise ValueError(f"duplicate restart variable {name}")
        if isinstance(obj, torch.Tensor):
            vals = array(obj)
            dims, units = _field_layout(name, vals)
            ds[name] = xr.Variable(dims, vals, attrs={"units": units} if units else {})
            return ["tensor", name, list(dims)]
        if type(obj) not in (float, int, bool):
            raise TypeError(f"unsupported restart field {name}: {type(obj)}")
        ds[name] = xr.Variable((), obj)
        return [type(obj).__name__, name]

    ds.attrs["state_structure"] = json.dumps(encode(payload, ()), allow_nan=False, separators=(",", ":"))
    attach_mesh_attributes(ds)
    write_netcdf(path, ds)


def read_checkpoint(path):
    """Recreate the original CPU tensor/primitive tree, including optional caches."""
    path = Path(path)
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True)
    if path.suffix != ".nc":
        raise ValueError("expected a .nc or .pt restart")
    with xr.open_dataset(path, engine="netcdf4", decode_times=False, mask_and_scale=False) as ds:
        if ds.attrs.get("vfs_format") != "coupled_restart_v1":
            raise ValueError("expected a VFS coupled restart")

        def decode(node):
            kind, item, *extra = node
            if kind == "dict":
                return {k: decode(v) for k, v in item.items()}
            if kind in ("list", "tuple"):
                seq = [decode(v) for v in item]
                return seq if kind == "list" else tuple(seq)
            if kind == "literal":
                return item
            var = ds[item]
            if kind == "tensor":
                if var.dims != tuple(extra[0]):
                    raise ValueError(f"restart dimensions differ for {item}")
                return torch.from_numpy(var.values.copy())
            return {"int": int, "float": float, "bool": bool}[kind](var.item())

        payload = decode(json.loads(ds.attrs["state_structure"]))
        if payload["fingerprint"] != ds.attrs["fingerprint"] or payload["air"]["time_seconds"] != ds.time.item():
            raise ValueError("inconsistent restart identity or clock")
        return payload


def append_diagnostics(path, row, adapter, fingerprint):
    """Close the time-series file after each sample so live readers can reopen it."""
    from netCDF4 import Dataset

    def units(name):
        for suffix, unit in (("_J_m2", "J m-2"), ("_kg_m2_global", "kg m-2"),
                             ("_kg_m2", "kg m-2"), ("_J_kg", "J kg-1"), ("_m_s", "m s-1"),
                             ("_kg", "kg"), ("_K", "K"), ("_m", "m")):
            if name.endswith(suffix):
                return unit
        return None

    def append(ds):
        index = len(ds.dimensions["time"])
        for name, value in row.items():
            if name != "seconds":
                ds[name][index] = np.nan if value is None else value
        ds["time"][index] = row["seconds"]

    if not path.exists():
        with atomic_path(path) as tmp, Dataset(tmp, "w", format="NETCDF4") as ds:
            ds.vfs_format = "coupled_diagnostics_v1"
            ds.fingerprint = fingerprint
            ds.createDimension("time", None)
            ds.createVariable("time", "i8", ("time",)).setncatts(clock_attrs(adapter))
            for name in row:
                if name != "seconds":
                    var = ds.createVariable(name, "f8", ("time",))
                    if units(name):
                        var.units = units(name)
            append(ds)
    else:
        with Dataset(path, "a") as ds:
            append(ds)
