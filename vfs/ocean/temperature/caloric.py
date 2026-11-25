"""Surface caloric law and analytic entropy means for constant heat capacity."""
import torch


class ConstantHeatCapacity:
    """Idealized constant-cp Gibbs reference, independent of salinity.

    h0=cp*T90; eta=cp*log((T90+273.15)/273.15), so dh0/deta=Kelvin.
    This is not TEOS seawater calorics.
    """
    temperature_scale='IPTS-68'
    to_90=1/1.00024
    data={'temperature_ITS90_range_C':[-2.5,40.],'practical_salinity_range':[0.,42.]}

    def __init__(self,config):
        self.cp=config.cp;self.cp_theta=self.cp*self.to_90

    def evaluate(self,sp,theta,*,entropy_derivatives=False):
        if sp.dtype!=theta.dtype or sp.device!=theta.device:raise ValueError('T/S dtype and device must match')
        t=self.to_90*theta
        if not torch.isfinite(sp).all() or not torch.isfinite(t).all():raise ValueError('finite T/S required')
        if ((sp<0)|(sp>42)|(t<-2.5)|(t>40)).any():raise ValueError('outside linear reference domain')
        return self._values(theta, t, entropy_derivatives=entropy_derivatives)

    def _values(self,theta,t,*,entropy_derivatives=False):
        """Pure tensor calorics; evaluate retains the public domain checks."""
        kelvin=t+273.15;eta=self.cp*torch.log1p(t/273.15);zero=torch.zeros_like(theta)
        result=dict(h0=self.cp*t,entropy=eta,cp=zero+self.cp,h_theta=zero+self.cp_theta,h_s=zero,
            entropy_theta=self.cp_theta/kelvin,kelvin=kelvin,gibbs=self.cp*t-kelvin*eta)
        if entropy_derivatives:result['entropy_s']=zero
        return result

    def temperature_from_enthalpy(self,sp,enthalpy):
        theta=enthalpy/self.cp_theta;self.evaluate(sp,theta);return theta

    def inverse_entropy(self,sp,entropy,*,guess=None):
        if guess is not None and (guess.shape!=entropy.shape or guess.dtype!=entropy.dtype or guess.device!=entropy.device):
            raise ValueError('temperature guess must match entropy')
        theta=273.15*torch.expm1(entropy/self.cp)/self.to_90
        self.evaluate(sp,theta,entropy_derivatives=True)
        return theta

    def face_means(self,si,ti,sj,tj):
        vi=self.evaluate(si,ti,entropy_derivatives=True);vj=self.evaluate(sj,tj,entropy_derivatives=True)
        # eta_face = integral_0^1 eta(Ti+s*(Tj-Ti)) ds = -Delta G/Delta Kelvin.
        mean=.5*(vi['kelvin']+vj['kelvin']);x=(vj['kelvin']-vi['kelvin'])/(2*mean)
        close=x.abs()<1e-3;den=torch.where(close,torch.ones_like(x),x)
        correction=((1+x)*torch.log1p(x)-(1-x)*torch.log1p(-x))/(2*den)-1
        series=-x.square()/6-x.pow(4)/20-x.pow(6)/42-x.pow(8)/72
        eta=self.cp*(torch.log(mean/273.15)+torch.where(close,series,correction))
        salt=.5*(si+sj);zero=torch.zeros_like(eta)
        dg=vj['gibbs']-vi['gibbs'];work=eta*(vj['kelvin']-vi['kelvin'])
        return eta,salt,dict(identity=dg+work,scale=dg.abs()+work.abs(),chemical_susceptibility=zero,composition_entropy=zero)
