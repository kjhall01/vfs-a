# Contributing

Development is on [GitHub](https://github.com/kjhall01/vfs-a). Read
[AGENTS.md](AGENTS.md), [the handoff](handoff/index.md),
[pass-4 results](docs/PASS4_RESULTS.md), and the
[physical-fields guide](docs/PHYSICAL_FIELDS_API.md) before changing behavior.

## Local setup

Use Python 3.12 to match CI. From the checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install build
# Linux CPU baseline build:
python -m pip install 'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install --prefer-binary -e '.[test]'
python -m pytest tests --fixture-mode=portable -q
```

The `test` extra adds only pytest to the existing runtime requirements; the
large `dev` extra and `environment.yml` are unnecessary for these checks.
`--prefer-binary` avoids compiling dependencies such as netCDF4 when a suitable
released wheel is available.
On macOS use `torch==2.15.0` in the PyTorch install command (no `+cpu` wheel).
Other platforms/builds are compatibility checks, not promises of exact equality.
The [public CPU index](https://download.pytorch.org/whl/cpu/torch/) lists the
Linux Python 3.12 `2.15.0+cpu` wheel, checked on 2026-10-29.

## Test the installed wheel

Use a fresh environment so an editable install cannot mask packaging errors.
With the setup environment active, run from the checkout (Bash):

```bash
check_dir=$(mktemp -d)
python -m build --wheel --outdir "$check_dir/wheels"
python -m venv "$check_dir/venv"
source "$check_dir/venv/bin/activate"
python -m pip install 'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
wheels=("$check_dir"/wheels/*.whl)
python -m pip install --prefer-binary "${wheels[0]}[test]"
python -m pip check
cp -R tests tools examples "$check_dir/"
cd "$check_dir"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -I -c 'import vfs, torch; print(vfs.__file__, torch.__version__)'
python -I -m pytest tests --fixture-mode=portable -q
python -I tools/check_baseline.py --mode portable
python -I tools/check_dry_helpers.py --mode portable
python -I examples/moist_atmosphere.py
python -I examples/bounded_moist.py
python -I examples/held_suarez.py
python -I examples/dry_atmosphere.py
python -I examples/dry_atmosphere.py --imex
python -I examples/moist_atmosphere.py --imex
```

Apply the same macOS version substitution here. Confirm the printed package
path is in the new environment's `site-packages`. CI copies only tests, tools
and examples outside the checkout and asserts the installed import path.
The checks run separately so a failing comparison does not hide smoke results.

Portable mode is the required CI gate, including both fixture tests inside
pytest and both standalone tools. The [policy](docs/PORTABLE_REGRESSION.md)
explains the explicit quantity mapping, regression budgets and independent
invariants. Exact mode remains the default unless selected explicitly:

```bash
python -I -m pytest tests --fixture-mode=exact -q
python -I tools/check_baseline.py --mode exact
python -I tools/check_dry_helpers.py --mode exact
python -I tools/record_environment.py --report environment.json
```

CI runs exact checks separately as informational steps, retaining their failing
exit status in reports and the job summary. A portable pass is not an exact pass.
See [validation results](docs/PORTABLE_REGRESSION_RESULTS.md) for observed outcomes.

Fixtures record CPU float64 with PyTorch `2.15.0+cpu`. Exact comparisons cover
142 checkpoint-54 and 93 retained conservative dry-helper tensors. The original
dry fixture also archives 100 deliberately retired canonical-core tensors; see
[the removal and validation record](docs/READABILITY_RESULTS.md). Do not regenerate fixtures (`--reference` / `--capture`) or loosen
tolerances to fix CI. Report version, platform, failing fields and errors;
investigate differences before proposing numerical changes. A matching torch
version alone does not guarantee bitwise agreement across platforms.

## Prepare a focused PR

Create a topic branch and keep one concern per PR. Preserve numerical defaults,
state conventions, interface/edge staggering and integrators. MoistAtmosphere
uses `BoundedGibbs(base)` and its bounded Euler species
amount ledger. Vapor must remain strictly positive, with no floor; exact-zero
water equivalence to the dry core is not established.

Run the wheel checks above, review `git diff --check`, and update the handoff
and relevant results documentation. Describe the behavior changed, local
environment/results, remaining failures, and actual GitHub Actions results
separately (link the run when available). CPU float64 forward checks do not
qualify GPU, long runs or full backward rollouts. Keep numerical changes and
scientific verification separate. Do not publish packages or select a license
as part of routine contributor setup.

Package scope and deliberate API removals are documented in
[current paths](docs/CURRENT_PATHS.md) and [the removal record](docs/READABILITY_RESULTS.md). Do not restore removed
domain/ML APIs merely to satisfy historical compatibility expectations.

Before numerical changes, use `tools/capture_readability.py --save /tmp/before.pt`
with the baseline installed wheel, then `--compare /tmp/before.pt` with the changed
wheel in the same environment. It checks retained states, energies, gradients,
ledgers, ARK4 stages and solver records exactly; it never replaces fixtures.
See [the Held–Suarez archive command](docs/HELD_SUAREZ.md#reproduce-the-archive-comparison).
The next phase is [scientific verification](docs/VERIFICATION_HANDOFF.md), with the
implementation frozen; Conda packaging is deferred.
