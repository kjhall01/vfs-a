"""Fixed-step Kennedy--Carpenter additive RK pairs, using the existing solver.

Coefficients: SUNDIALS Butcher Tables, ARK324L2SA and ARK436L2SA ERK/DIRK:
https://sundials.readthedocs.io/en/latest/arkode/Butcher_link.html
Kennedy & Carpenter (2003), Applied Numerical Mathematics 44, 139--181.
Only the main common weights are used; no embedded adaptivity is implied.
These methods are NOT positivity-preserving SSP extensions of SSPRK3.
"""
from .ark2 import ARKTableau, ARK2_TABLEAU, DryPEIMEXARK2

_G3 = 1767732205903 / 4055673282236
_B3 = (1471266399579/7840856788654, -4482444167858/7529755066697,
       11266239266428/11593286722821, _G3)
ARK3_TABLEAU = ARKTableau(
    explicit=((0.,0.,0.,0.), (2*_G3,0.,0.,0.),
              (5535828885825/10492691773637,788022342437/10882634858940,0.,0.),
              (6485989280629/16251701735622,-4246266847089/9704473918619,
               10755448449292/10357097424841,0.)),
    implicit=((0.,0.,0.,0.), (_G3,_G3,0.,0.),
              (2746238789719/10658868560708,-640167445237/6845629431997,_G3,0.),_B3),
    weights=_B3, order=3, name='ark3-kennedy-carpenter-4stage')
_B4 = (82889/524892,0.,15625/83664,69875/102672,-2260/8211,1/4)
ARK4_TABLEAU = ARKTableau(
    explicit=((0.,0.,0.,0.,0.,0.), (1/2,0.,0.,0.,0.,0.),
              (13861/62500,6889/62500,0.,0.,0.,0.),
              (-116923316275/2393684061468,-2731218467317/15368042101831,
               9408046702089/11113171139209,0.,0.,0.),
              (-451086348788/2902428689909,-2682348792572/7519795681897,
               12662868775082/11960479115383,3355817975965/11060851509271,0.,0.),
              (647845179188/3216320057751,73281519250/8382639484533,
               552539513391/3454668386233,3354512671639/8306763924573,4040/17871,0.)),
    implicit=((0.,0.,0.,0.,0.,0.), (1/4,1/4,0.,0.,0.,0.),
              (8611/62500,-1743/31250,1/4,0.,0.,0.),
              (5012029/34652500,-654441/2922500,174375/388108,1/4,0.,0.),
              (15267082809/155376265600,-71443401/120774400,
               730878875/902184768,2285395/8070912,1/4,0.),_B4),
    weights=_B4, order=4, name='ark4-kennedy-carpenter-6stage')
TABLEAUX = {'ark2':ARK2_TABLEAU,'ark3':ARK3_TABLEAU,'ark4':ARK4_TABLEAU}


class DryPEIMEXARK(DryPEIMEXARK2):
    """Select a fixed-step ARK pair while reusing the same frozen wave solver.

    All nonlinear evaluations, split conventions, residual checks and Schur
    solvers are inherited unchanged. Old DryPEIMEXARK2 behavior is untouched.
    Neither high order nor implicit waves imply positivity or exact finite-step
    nonlinear energy conservation. Moist states are not supported here.
    """
    def __init__(self, dynamics, solver_config=None, *, method='ark2',
                 explicit_tendency_providers=None):
        if method not in TABLEAUX:
            raise ValueError(f'Unknown ARK method {method!r}; choose {tuple(TABLEAUX)}')
        super().__init__(dynamics,solver_config,
                         explicit_tendency_providers=explicit_tendency_providers)
        self.tableau=TABLEAUX[method]
