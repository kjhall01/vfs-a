"""Explicit regression versions; historical files and tolerance schema stay fixed."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1] / 'tests/fixtures'
VERSIONS = ('historical-v1', 'corrected-gibbs-v1')
PROVENANCE_SHA256 = 'cafddf4151a478c0caf4da767f7fb0da879eb24547e35a450de05878a8364037'


def reference_path(version):
    if version not in VERSIONS:
        raise ValueError('Unknown regression reference version: ' + version)
    if version == 'historical-v1':
        return ROOT / 'checkpoint54.pt'
    folder = ROOT / version
    raw = (folder / 'provenance.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROVENANCE_SHA256:
        raise ValueError('Corrected reference provenance hash differs')
    provenance = json.loads(raw)
    path = folder / 'corrected_gibbs_v1.pt'
    if hashlib.sha256(path.read_bytes()).hexdigest() != provenance['files'][path.name]:
        raise ValueError('Corrected reference fixture hash differs')
    if hashlib.sha256((ROOT/'checkpoint54.pt').read_bytes()).hexdigest() != provenance['historical_sha256']:
        raise ValueError('Historical reference hash differs')
    return path


def load_reference(version):
    return torch.load(reference_path(version), map_location='cpu', weights_only=True)
