#!/usr/bin/env bash
# Launch the coupled 4,096-cell Earth experiment from a VFS checkout.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
exec python -u -m workflows.coupled.campaign \
    --config workflows/coupled/earth_shelf_4096.yaml \
    --source "${PYAM21_SOURCE:-$repo/../PyAM21}" \
    --mesh "${VFS_MESH:-$repo/input-data/mesh_4096_exact_3000.npy}" \
    --geography "${VFS_GEOGRAPHY:-$repo/input-data/earth_4096_geography.npz}" \
    --radiation-data "${PYAM21_RADIATION_DATA:-$repo/input-data/sea_esf}" \
    --verbose "$@"
