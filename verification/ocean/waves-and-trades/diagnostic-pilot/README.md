# Retained diagnostic pilot

The first 256-cell, dt=60 s wave integration completed with the same dynamics
as the final repeated run. Its newly added elevation diagnostics accidentally
broadcast column-shaped eta against vector-shaped cell areas. It also lacked
intermediate snapshots. It is retained here but excluded from the seven-run
comparison. The run was repeated with shape-safe diagnostics and snapshots.
No model method, coefficient, or timestep was changed to obtain completion.
The other original runs retain raw CSV/summary diagnostics; the analysis
recomputes corrected elevation RMS and initial-pattern projection from their
saved snapshots. Energy, inventory and native-speed diagnostics were unaffected.
