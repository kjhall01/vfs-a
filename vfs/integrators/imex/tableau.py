"""Kennedy--Carpenter ARK436L2SA, six stages, fourth order.

Coefficients unchanged from the retained implementation. Common main weights;
no embedded adaptivity. ARK4 is not an SSP or species-bounded method.
"""
from typing import NamedTuple


class ARKTableau(NamedTuple):
    explicit: tuple[tuple[float, ...], ...]
    implicit: tuple[tuple[float, ...], ...]
    weights: tuple[float, ...]
    order: int
    name: str

    @property
    def n_stages(self) -> int:
        return len(self.weights)


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
