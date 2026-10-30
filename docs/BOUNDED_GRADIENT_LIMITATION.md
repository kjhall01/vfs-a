# Pre-existing bounded gradient failure

Reproduced on the pre-change PR #5 installed wheel and again after the module
integration, macOS arm64 / PyTorch 2.15.0 / CPU float64. Autograd was enabled;
no tensor was detached and no numerical operator was changed to avoid the error.

Minimal valid two-node, one-face transfer:

```python
import torch
from vfs.atmosphere.bounded_transport import bounded_transfer

m = torch.ones(2, dtype=torch.float64)
q = torch.tensor([[.9, .1, 0., 0.], [.9, .1, 0., 0.]], dtype=torch.float64)
g = torch.tensor([.1], dtype=torch.float64, requires_grad=True)
j = g[:, None] * q[:1]
bounded_transfer(m, q, torch.tensor([0]), torch.tensor([1]), g, j)
# TypeError: allclose(): argument 'atol' must be float, not Tensor
```

The unchanged assertion uses `atol=128 * eps * scale`, where
`scale = G.abs().max().clamp_min(torch.finfo(G.dtype).tiny)`. When G carries a
gradient, this tolerance is a gradient-carrying scalar tensor. This environment's
allclose argument conversion rejects it. Ordinary nongradient forward calls pass.
This is a validation/API failure before the transfer result, not evidence that
the underlying derivative is numerically valid or invalid.

## Workflow probes

A 16-cell, three-layer CPU float64 state used T=280 K, ps=100000 Pa, wind=1 m/s,
vapor/liquid/ice=.01/.002/.001 and dt=1 s. Each row enabled gradients on only
that prognostic tensor. Successful derivatives used scalar energy or summed
output entropy, checking finiteness only, not derivative accuracy.

| Gradient input | Energy diagnostic | Bounded evaluate / Euler | Bounded SSPRK3 | Centered SSPRK3 |
| --- | --- | --- | --- | --- |
| Wind | finite derivative | TypeError | TypeError | finite one-step derivative |
| Surface pressure | finite derivative | TypeError | TypeError | finite one-step derivative |
| Entropy content | finite derivative | finite derivative | TypeError | finite one-step derivative |
| Water content | finite derivative | finite derivative | TypeError | finite one-step derivative |

Entropy/composition feed momentum and eventually carrier flux, so later bounded
stages also encounter the failing tolerance conversion. Consequently the ordinary
bounded facade does not currently support backpropagation through even this
one-step probe with any of these initial-state gradients. Gradient-preserving
initialization, diagnostics and explicit state conversion remain usable.
Centered one-step success is not long-rollout or IMEX backward qualification.

This PR deliberately leaves the assertion unchanged. Any fix belongs in a separate
narrow change with a review of the assertion's comparison semantics and gradient
behavior. No detach, no_grad workaround, tolerance relaxation or fixture changes
are proposed here to label blocked workflows as passing.
