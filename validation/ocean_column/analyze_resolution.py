"""Summarize completed campaign files without uploading oversized raw profiles."""
import argparse, hashlib, json
from pathlib import Path
from analyze_sustained import compare


def summarize(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    raw = {}
    manifest = []
    for path in sorted([*source.glob('L*.json'), *source.glob('mesh*.json')]):
        data = json.loads(path.read_text())
        if isinstance(data.get('profiles'), list):
            raw[path.stem] = data
            summary = {k: v for k, v in data.items() if k != 'profiles'}
            summary.update(initial_profile=data['profiles'][0], final_profile=data['profiles'][-1],
                hourly=[{k: p[k] for k in ('time_s', 'energy_minus_input_J_kg', 'temperature_mld_m', 'limiter')} for p in data['profiles']])
        else:
            summary = data
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        summary['raw_sha256'] = digest
        text = json.dumps(summary, indent=2) + '\n'
        if len(text.encode()) > 131072:
            raise ValueError(f'compact summary exceeds cap: {path.name}')
        (destination / path.name).write_text(text)
        manifest.append(dict(file=path.name, raw_bytes=path.stat().st_size, raw_sha256=digest,
            raw_omitted=path.stat().st_size > 131072,
            representation='endpoint profiles and hourly scalar diagnostics' if isinstance(data.get('profiles'), list) else 'complete metrics'))
    comparisons = []
    def add(left, right, kind):
        if left not in raw or right not in raw:
            return
        a, b = raw[left], raw[right]
        if a['status'] != 'completed' or b['status'] != 'completed' or a['completed_hours'] != b['completed_hours']:
            return
        comparisons.append(dict(left=left, right=right, kind=kind, metrics=compare(a, b)))
    for layers in (32, 64):
        for suffix in ('base', 'sub8', 'dt150', 'dt75'):
            add(f'L{layers}-{suffix}', f'L{layers}-reference75', 'numerical_reference')
        add(f'L{layers}-factor05', f'L{layers}-base', 'safety_factor')
    for suffix in ('base', 'dt75', 'wind24'):
        add('L32-' + suffix, 'L64-' + suffix, 'vertical_overlap')
    add('L64-dt75', 'L128-dt75', 'vertical_overlap')
    (destination / 'comparisons.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    summarize(a.input, a.output)
