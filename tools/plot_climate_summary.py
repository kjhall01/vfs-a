"""Plot latitude–pressure sections and native surface maps from coupled NetCDF files."""
import argparse
from pathlib import Path

from vfs_plotting import plot_sections, plot_surface, summarize


def main():
    from matplotlib import pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="one or more snapshot or restart .nc files")
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument("--days", nargs=2, type=float, metavar=("START", "END"), help="inclusive simulated days")
    args = parser.parse_args()
    summary = summarize(args.files, args.days)
    args.output.mkdir(exist_ok=True, parents=True)
    for name, draw in [("circulation", plot_sections), ("surface", plot_surface)]:
        fig = draw(summary)
        path = args.output / (name + ".png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(path)
    if "precipitation" not in summary:
        print("These states contain no saved precipitation flux. The moisture panel shows column water vapor.")


if __name__ == "__main__":
    main()
