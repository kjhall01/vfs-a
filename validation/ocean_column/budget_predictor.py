"""Research-only analytic energy-cost predictor; not a calibrated closure.

At fixed h, pair exchange c transfers c*Delta(SP) and c*Delta(h0).
The reference enthalpy transfer cancels in the pair energy sum. The remaining
linear work is computed from the matched JMD95 parcel-potential derivatives.
Actual nonlinear work, TKE affordability and entropy are still checked by the
installed TKEClosure; its original backtracking remains the fallback.
"""
from dataclasses import dataclass
import math
import torch
from vfs.ocean.stratified import TKEClosure


def pair_work_slope(ocean, physical, reference):
    """dW/dc at c=0 [J/m]; c has units m. No forward autograd."""
    base=ocean.energy.base
    h=physical.h
    node,weight=base._quadrature(h)
    z=base.height(physical)[...,None]+h[...,None]*(node-.5)
    _,pt,ps,_=base.parcel_potential(physical.temperature()[...,None],
                                   physical.salinity()[...,None],z)
    pt=(pt*weight).sum(-1);ps=(ps*weight).sum(-1)
    # dtheta = (dh0 - h_s*dSP)/h_theta.
    a=pt/reference['h_theta']
    b=ps-a*reference['h_s']
    ds=physical.salinity()[:,1:]-physical.salinity()[:,:-1]
    dh=reference['h0'][:,1:]-reference['h0'][:,:-1]
    return ocean.energy.mass_area(physical)*((a[:,:-1]-a[:,1:])*dh+(b[:,:-1]-b[:,1:])*ds)


@dataclass(frozen=True)
class BudgetPredictor(TKEClosure):
    """Predict an affordable first trial, then retain all production guards.

    The margin is a research parameter, not evidence of a physical efficiency.
    Nonpositive linear cost leaves the proposal unchanged: nonlinear work can
    still be positive, so no acceptance decision is based on this derivative.
    """
    margin: float = .95

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.margin) or not 0<self.margin<=1:
            raise ValueError('predictor margin must be in (0, 1]')

    def _mixing_proposal(self,ocean,physical,reference,reservoir,proposed):
        h=physical.h;alpha=h[:,:-1]/(h[:,:-1]+h[:,1:])
        budget=torch.minimum(reservoir[:,:-1]/alpha,reservoir[:,1:]/(1-alpha))
        slope=pair_work_slope(ocean,physical,reference)
        cost=proposed*slope
        limited=(slope>0)&(cost>budget)
        affordable=self.margin*budget/torch.where(slope>0,slope,torch.ones_like(slope))
        return torch.where(limited,torch.minimum(proposed,affordable),proposed)
