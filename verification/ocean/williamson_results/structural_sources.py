"""Exact, versioned runtime relocation map for the reviewed #25 tree.

Only this pinned commit and every explicitly mapped file are accepted. This is
not an import-insensitive or wildcard source policy; unknown edits still fail.
"""
import hashlib
import json
from pathlib import Path

COMMIT = '6c5bf938503b357ba01c50ab99ae7e2cc770e90a'
MAP = Path(__file__).with_name('runtime_relocations_v1.json')
MAP_SHA256 = 'c957d6e3971a57f646e87cd1fd4543234e086996b50c1178ac919b8228453b2b'


NUMERICAL_COMMIT = "a7facc1bcdb88f7dc9c10f4fb6059f29b0a9bac7"
NUMERICAL_PATH = "vfs/atmosphere/gibbs_means.py"
# Reviewed ocean #24, including model-aware output. Only vfs/ocean is admitted
# from this exact commit; atmospheric/shared files retain the original map.
OCEAN_COMMIT = "8936334f027e5660987a15daa000685746ff5b47"


def verify(repo, package, revision, git, numerical_revision=None, ocean_revision=None):
    if ocean_revision not in (None, OCEAN_COMMIT):
        raise ValueError('Unknown ocean revision')
    if numerical_revision not in (None, NUMERICAL_COMMIT):
        raise ValueError("Unknown proposed numerical revision")
    if revision != COMMIT:
        raise ValueError('Unknown structural revision')
    raw = MAP.read_bytes()
    digest = lambda b: hashlib.sha256(b).hexdigest()
    if digest(raw) != MAP_SHA256:
        raise ValueError('Structural mapping hash differs')
    mapping = json.loads(raw)
    if mapping['structural_commit'] != COMMIT:
        raise ValueError('Structural mapping commit differs')
    paths = git(repo, 'ls-tree', '-r', '--name-only', COMMIT, 'vfs').decode().splitlines()
    if set(paths) != set(mapping['runtime']):
        raise ValueError('Structural mapping coverage differs')
    ocean_paths = []
    if ocean_revision:
        ocean_paths = git(repo, 'ls-tree', '-r', '--name-only', OCEAN_COMMIT,
                          'vfs/ocean').decode().splitlines()
        if not ocean_paths or any(not p.startswith('vfs/ocean/') for p in ocean_paths):
            raise ValueError('Pinned ocean tree is unavailable or invalid')
    hashes = {}
    for name in paths:
        row = mapping['runtime'][name]
        expected = git(repo, 'show', f'{COMMIT}:{name}')
        if digest(expected) != row['sha256']:
            raise ValueError(f'Structural source hash differs: {name}')
        for old, checksum in row['old_sha256'].items():
            if digest(git(repo, 'show', f"{mapping['before_structure']}:{old}")) != checksum:
                raise ValueError(f'Relocation origin differs: {old}')
        if numerical_revision and name == NUMERICAL_PATH:
            expected = git(repo, 'show', f'{NUMERICAL_COMMIT}:{name}')
        if ocean_revision and name.startswith('vfs/ocean/'):
            if name not in ocean_paths:
                raise ValueError(f'Pinned ocean tree omits mapped file: {name}')
            expected = git(repo, 'show', f'{OCEAN_COMMIT}:{name}')
        installed = package / Path(name).relative_to('vfs')
        if not installed.is_file() or installed.read_bytes() != expected:
            raise ValueError(f'Installed runtime differs from structural mapping: {name}')
        hashes[name] = digest(expected)
    for name in ocean_paths:
        if name in hashes:
            continue
        expected = git(repo, 'show', f'{OCEAN_COMMIT}:{name}')
        installed = package / Path(name).relative_to('vfs')
        if not installed.is_file() or installed.read_bytes() != expected:
            raise ValueError(f'Installed ocean source differs from pinned revision: {name}')
        hashes[name] = digest(expected)
    actual = {'vfs/' + str(p.relative_to(package)) for p in package.rglob('*.py')}
    if actual != {n for n in hashes if n.endswith('.py')}:
        raise ValueError('Installed runtime contains unexpected Python modules')
    return hashes
