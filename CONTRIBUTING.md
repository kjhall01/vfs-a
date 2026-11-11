# Contributing

See [package structure](docs/PACKAGE_OWNERSHIP.md), [state conventions](docs/STATE_OWNERSHIP.md)
and [verification](verification/README.md) before changing numerical behavior.

## Local setup

Use Python 3.12 to match CI. From the checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install build
# Linux CPU baseline build:
python -m pip install 'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install --prefer-binary -e '.[test]'
python -m pip install 'gsw==3.6.23'  # independent ocean verification reference
python -m pytest tests --fixture-mode=portable -q
```

The `test` extra adds only pytest to the existing runtime requirements; the
large `dev` extra and `environment.yml` are unnecessary for these checks.
`--prefer-binary` avoids compiling dependencies such as netCDF4 when a suitable
released wheel is available.
On macOS use `torch==2.15.0` in the PyTorch install command (no `+cpu` wheel).
Other platforms/builds are compatibility checks, not promises of exact equality.

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
python -m pip install 'gsw==3.6.23'
python -m pip check
cp -R tests tools examples "$check_dir/"
cd "$check_dir"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -I -c 'import vfs, torch; print(vfs.__file__, torch.__version__)'
python -I -m pytest tests --fixture-mode=portable -q
python -I tools/check_baseline.py --reference-version corrected-gibbs-v1 --mode portable
python -I tools/check_dry_helpers.py --mode portable
python -I examples/moist_atmosphere.py
python -I examples/bounded_moist.py
python -I examples/held_suarez.py
python -I examples/dry_atmosphere.py
python -I examples/dry_atmosphere.py --imex
python -I examples/moist_atmosphere.py --imex
python -I examples/ocean_stratified.py
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
python -I tools/check_baseline.py --reference-version corrected-gibbs-v1 --mode exact
python -I tools/check_dry_helpers.py --mode exact
python -I tools/record_environment.py --report environment.json
```

CI runs exact checks separately as informational steps, retaining their failing
exit status in reports and the job summary. A portable pass is not an exact pass.
See [validation results](docs/PORTABLE_REGRESSION_RESULTS.md) for observed outcomes.

Historical fixtures record CPU float64 with PyTorch `2.15.0+cpu`; the separately
versioned corrected Gibbs reference records macOS CPU float64, PyTorch `2.15.0`.
See [reference provenance and reproduction](verification/ATMOSPHERIC_INTEGRATION.md).
The standalone baseline tool defaults to the historical reference to preserve
old commands; select `--reference-version corrected-gibbs-v1` for current code.
Historical disagreement is evidence of the changed operator, not a current-code
regression. Exact comparisons cover
142 checkpoint-54 and 93 retained conservative dry-helper tensors. The original
dry fixture also archives 100 deliberately retired canonical-core tensors; see
[the removal and validation record](docs/READABILITY_RESULTS.md). Do not regenerate fixtures (`--reference` / `--capture`) or loosen
tolerances to fix CI. Report version, platform, failing fields and errors;
investigate differences before proposing numerical changes. A matching torch
version alone does not guarantee bitwise agreement across platforms.

## Numerical changes

Preserve documented state conventions and explicitly distinguish algorithm changes
from refactoring. Compare states, stage ledgers, energy/work and gradients where
relevant. Do not regenerate fixtures or loosen tolerances to resolve failures.
Keep verification artifacts and source hashes, and report actual tested scope.

Use the portable suite as the required regression check. Run relevant standalone
examples and same-environment captures for the affected component. The full CI
workflow also runs pinned historical experiments and retains informational exact
fixture discrepancies. See [the workflow](.github/workflows/ci.yml).

Keep changes focused and run `git diff --check`. Document behavior, units,
configuration and limits in the user guides; record detailed measurements in
verification reports. CPU float64 checks do not imply accelerator, long-run or
full backward support.
