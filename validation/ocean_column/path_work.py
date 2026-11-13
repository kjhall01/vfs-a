"""Cancellation-safe JMD95 mixing work along conserved (SP,h0) paths.

Research only. Analytic thermodynamic derivatives, no forward autograd. Preserve
both the parent's midpoint energy flux and its local allocation: the difference
between path-averaged and midpoint conjugates is retained as a flux divergence.
Full actual-H, entropy, positivity and nonlinear-solve checks are unchanged.
"""
from dataclasses import dataclass
import numpy as np
import torch
from coupled import CoupledClosure


def conjugates(ocean, temperature, salinity, z, weights):
    reference=ocean.reference.evaluate(salinity,temperature)
    _,pt,ps,_=ocean.energy.base.parcel_potential(temperature[...,None],salinity[...,None],z)
    a=(pt*weights).sum(-1)/reference['h_theta']
    b=(ps*weights).sum(-1)-a*reference['h_s']
    return a,b


@dataclass(frozen=True)
class PathWorkClosure(CoupledClosure):
    quadrature_order: int = 3

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.quadrature_order,int) or not 1<=self.quadrature_order<=16:
            raise ValueError('path quadrature order must be an integer in [1,16]')
        x,w=np.polynomial.legendre.leggauss(self.quadrature_order)
        object.__setattr__(self,'_path_nodes',tuple((x+1)/2))
        object.__setattr__(self,'_path_weights',tuple(w/2))

    def _buoyancy_work(self,ocean,before,after,enthalpy_flux,salt_flux,default_work):
        h=before.h;area=ocean.energy.mass_area(before)
        zero=torch.zeros_like(h[:,:1])
        div=lambda f:torch.cat((f,zero),-1)-torch.cat((zero,f),-1)
        ds=div(salt_flux)/h;dh=div(enthalpy_flux)/h
        s0=before.salinity();t0=before.temperature()
        H0=ocean.reference.evaluate(s0,t0)['h0']
        base=ocean.energy.base;node,weight=base._quadrature(h)
        z=base.height(before)[...,None]+h[...,None]*(node-.5)
        a_bar=torch.zeros_like(h);b_bar=torch.zeros_like(h)
        for fraction,w in zip(self._path_nodes,self._path_weights):
            salinity=s0+fraction*ds
            temperature=ocean.reference.temperature_from_enthalpy(salinity,H0+fraction*dh)
            a,b=conjugates(ocean,temperature,salinity,z,weight)
            a_bar=a_bar+w*a;b_bar=b_bar+w*b
        # Exactly the parent's midpoint conjugates (including actual reconstructed
        # endpoint fields). Changing these would change local work allocation.
        a_mid,b_mid=conjugates(ocean,.5*(t0+after.temperature()),.5*(s0+after.salinity()),z,weight)
        alpha=h[:,:-1]/(h[:,:-1]+h[:,1:])
        face=lambda q:(1-alpha)*q[:,:-1]+alpha*q[:,1:]
        interface=area*((a_bar[:,:-1]-a_bar[:,1:])*enthalpy_flux+(b_bar[:,:-1]-b_bar[:,1:])*salt_flux)
        paired=torch.cat((alpha*interface,zero),-1)+torch.cat((zero,(1-alpha)*interface),-1)
        correction=area*(face(a_bar-a_mid)*enthalpy_flux+face(b_bar-b_mid)*salt_flux)
        # No subtraction of finite parcel energies; all terms scale with flux.
        return paired+div(correction)
