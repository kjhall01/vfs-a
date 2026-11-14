"""Compare local tensor captures from independent field_reuse.py wheel samples."""
import argparse
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-roundoff', action='store_true')
    args = parser.parse_args()
    a = torch.load(args.parent.with_suffix('.pt'), weights_only=True)
    b = torch.load(args.candidate.with_suffix('.pt'), weights_only=True)
    if a.keys() != b.keys():
        raise AssertionError('state/report tensor structure changed')
    different = []
    fields = {}
    for key, left in a.items():
        right = b[key]
        if not torch.equal(left, right):
            different.append(key)
        if not torch.isfinite(right).all():
            raise AssertionError('nonfinite tensor: '+key)
        if left.dtype == torch.bool:
            continue
        delta = (right-left).double()
        fields[key] = dict(max_abs=delta.abs().max().item() if delta.numel() else 0.,
                          relative_l2=(delta.norm()/left.double().norm().clamp_min(1e-300)).item())
    result = dict(parent=args.parent.name, candidate=args.candidate.name,
                  tensor_count=len(a), bitwise_equal=not different,
                  different_tensors=different, field_errors=fields)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'field_errors'}))
    if different and not args.allow_roundoff:
        raise AssertionError('default path differs from parent; see comparison output')
    # --allow-roundoff records differences; it is not a numerical acceptance test.


if __name__ == '__main__':
    main()
