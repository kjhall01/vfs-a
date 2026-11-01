"""Plot saved Williamson measurements; does not rerun or modify the solver."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.results.read_text())
    cases = report['cases']
    dt_values = sorted(cases[0]['runs'], key=int, reverse=True)
    finest_dt = dt_values[-1]
    spacing = [case['mesh_spacing_m'] / 1000 for case in cases]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, key, label in zip(axes[0], ['h_error_m', 'u_error_m_s'],
                             ['Thickness RMS error (m)', 'Edge-normal velocity RMS error (m/s)']):
        for dt in dt_values:
            ax.loglog(spacing, [case['runs'][dt]['series'][-1][key]['l2'] for case in cases],
                      'o-', label=f'dt = {dt} s')
        ax.set(xlabel='Mean mesh spacing (km)', ylabel=label)
        ax.grid(alpha=.2, which='both')
        ax.legend()
    for term in ('pressure', 'kinetic', 'rotation'):
        axes[1, 0].loglog(spacing,
            [case['initial_balance']['term_errors_m_s2'][term]['l2'] for case in cases],
            'o-', label=f'{term} error')
    axes[1, 0].loglog(spacing,
        [case['initial_balance']['momentum_residual_m_s2']['l2'] for case in cases],
        's--', color='black', label='total RHS residual')
    axes[1, 0].set(xlabel='Mean mesh spacing (km)', ylabel='Initial acceleration RMS (m/s²)')
    axes[1, 0].grid(alpha=.2, which='both')
    axes[1, 0].legend()
    for case in cases:
        series = case['runs'][finest_dt]['series']
        axes[1, 1].plot([row['time_s'] / 86400 for row in series],
                       [row['h_error_m']['l2'] for row in series], 'o-', label=f"{case['cells']} cells")
    axes[1, 1].set(xlabel='Time (days)', ylabel='Thickness RMS error (m)',
                   title=f'Daily samples, dt = {finest_dt} s')
    axes[1, 1].grid(alpha=.2)
    axes[1, 1].legend()
    days = report['configuration']['days']
    fig.suptitle('Williamson case 2: frozen constant-density VFS ocean\n'
                 f'Axis-aligned flow · one layer · {days} days · no added damping', fontsize=13)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
