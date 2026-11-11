"""Research tensor surface Gibbs reference fitted to GSW-3.6.23 evaluations.

Only Python/torch in the forward path. This is a bounded-domain surface caloric
reference, not a replacement density EOS, not the full TEOS-10 Gibbs function.
SP uses reference composition; temperature scale is an explicit caller choice.
"""
import json
from pathlib import Path

import torch


def polynomial(x, coefficients):
    """Value and analytic first/second derivatives of an ordinary polynomial."""
    value = torch.zeros_like(x)+coefficients[-1]
    first, second = torch.zeros_like(x), torch.zeros_like(x)
    for c in reversed(coefficients[:-1]):
        second = second*x+2*first
        first = first*x+value
        value = value*x+c
    return value, first, second


class SurfaceGibbs:
    def __init__(self, coefficients, *, temperature_scale):
        if coefficients['format'] != 'surface-gibbs-reference-v1':
            raise ValueError('unsupported reference format')
        if temperature_scale not in ('IPTS-68', 'ITS-90'):
            raise ValueError('temperature_scale must be IPTS-68 or ITS-90')
        self.data = coefficients
        self.temperature_scale = temperature_scale
        self.to_90 = 1/1.00024 if temperature_scale == 'IPTS-68' else 1.

    @classmethod
    def from_file(cls, path, *, temperature_scale):
        return cls(json.loads(Path(path).read_text()), temperature_scale=temperature_scale)

    def evaluate(self, sp, theta, *, entropy_derivatives=False):
        if sp.dtype != theta.dtype or sp.device != theta.device:
            raise ValueError('salinity and temperature must share dtype/device')
        t = self.to_90*theta
        d = self.data
        if not torch.isfinite(sp).all() or not torch.isfinite(t).all():
            raise ValueError('finite thermodynamic inputs required')
        if (sp < 0).any() or (sp > d['practical_salinity_range'][1]).any():
            raise ValueError('salinity outside the surface-reference domain')
        if (t < d['temperature_ITS90_range_C'][0]).any() or (t > d['temperature_ITS90_range_C'][1]).any():
            raise ValueError('temperature outside the surface-reference domain')
        if entropy_derivatives and (sp <= 0).any():
            raise ValueError('chemical/entropy salinity derivative requires positive salinity')
        sa = d['ups']*sp
        y = (t-d['temperature_midpoint'])/d['temperature_halfspan']
        g, gt, gtt = polynomial(y, d['pure_power_coefficients'])
        gt, gtt = gt/d['temperature_halfspan'], gtt/d['temperature_halfspan']**2
        gs, gst = torch.zeros_like(g), torch.zeros_like(g)
        for i, row in enumerate(d['salt_power_coefficients']):
            c, ct, ctt = polynomial(y, row)
            ct, ctt = ct/d['temperature_halfspan'], ctt/d['temperature_halfspan']**2
            # S^(1+i/2), not S*sqrt(S)^i: first derivatives of the value
            # remain finite at S=0 without 0*infinity in the computation graph.
            factor = sa.pow(1+.5*i)/d['sa_scale']**(.5*i)
            factor_s = (1+.5*i)*sa.pow(.5*i)/d['sa_scale']**(.5*i)
            g, gt, gtt = g+factor*c, gt+factor*ct, gtt+factor*ctt
            gs, gst = gs+factor_s*c, gst+factor_s*ct
        kelvin = t+273.15
        log_entropy = d['entropy_log_coefficient']*torch.special.xlogy(sa, sa/35.16504)
        # The term T*S*log(S) cancels identically in h=g-T*g_T. Retain that
        # cancellation analytically instead of subtracting singular terms.
        cp = -kelvin*gtt
        result = dict(h0=g-kelvin*gt, entropy=-gt-log_entropy, cp=cp,
            h_theta=cp*self.to_90, h_s=(gs-kelvin*gst)*d['ups'],
            entropy_theta=-gtt*self.to_90, kelvin=kelvin,
            gibbs=g+kelvin*log_entropy)
        if entropy_derivatives:
            result['entropy_s'] = (-gst-d['entropy_log_coefficient']*(torch.log(sa/35.16504)+1))*d['ups']
        return result

    def temperature_from_enthalpy(self, sp, enthalpy):
        """Bracketed analytic Newton solve; no state clipping or forward AD.

        The bracket is the declared reference rectangle. The enthalpy input
        itself is never altered. Twelve fixed iterations retain tensor graphs;
        a failed inversion or out-of-domain content is rejected explicitly.
        """
        lower = torch.full_like(enthalpy, self.data['temperature_ITS90_range_C'][0]/self.to_90)
        upper = torch.full_like(enthalpy, self.data['temperature_ITS90_range_C'][1]/self.to_90)
        hlo, hhi = self.evaluate(sp, lower)['h0'], self.evaluate(sp, upper)['h0']
        if not torch.isfinite(enthalpy).all() or ((enthalpy < hlo)|(enthalpy > hhi)).any():
            raise ValueError('enthalpy outside the surface-reference domain')
        theta = lower+(upper-lower)*(enthalpy-hlo)/(hhi-hlo)
        for _ in range(12):
            values = self.evaluate(sp, theta)
            error = values['h0']-enthalpy
            lower = torch.where(error < 0, theta, lower)
            upper = torch.where(error > 0, theta, upper)
            proposal = theta-error/values['h_theta']
            theta = torch.where((proposal >= lower)&(proposal <= upper), proposal, .5*(lower+upper))
        error = (self.evaluate(sp, theta)['h0']-enthalpy).abs()
        tolerance = 128*torch.finfo(enthalpy.dtype).eps*(enthalpy.abs()+4000.)
        if (error > tolerance).any():
            raise FloatingPointError('surface enthalpy inversion failed its numerical residual check')
        return theta
