"""Short prescribed-SST dynamics/AM2 integration; no radiation or land."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from workflows.pyam21.run_column import parser, run

if __name__ == "__main__":
    run(parser(), coupled=True)
