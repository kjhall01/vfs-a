"""Fit a small differentiable surface potential to GSW function evaluations.

Independent fitting code; no GSW implementation is copied or modified. The
potential, rather than separately fitted heat/entropy functions, preserves the
derivative identities. Verification uses independent off-grid GSW evaluations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import gsw
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from numpy.polynomial.chebyshev import chebvander, cheb2poly


def fit():
    if gsw.__version__ != '3.6.23':
        raise RuntimeError('reproduce this reference with gsw==3.6.23')
    ups, tmin, tmax, spmax = 35.16504/35., -2.5, 40., 42.
    samax, midpoint, halfspan = ups*spmax, (tmin+tmax)/2, (tmax-tmin)/2
    degree_s, degree_t, nodes = 6, 8, 40
    z = np.cos(np.pi*(np.arange(nodes)+.5)/nodes)
    x, y = np.meshgrid((z+1)/2, z, indexing='ij')
    x, y = x.ravel(), y.ravel()
    sa, t = samax*x*x, midpoint+halfspan*y
    tx, ty = chebvander(2*x-1, degree_s), chebvander(y, degree_t)
    salt = (sa[:, None, None]*tx[:, :, None]*ty[:, None, :]).reshape(len(sa), -1)
    logarithmic = (t+273.15)*sa*np.log(sa/35.16504)
    design = np.column_stack([ty, salt, logarithmic])
    target = gsw.gibbs(0, 0, 0, sa, t, np.zeros_like(t))
    scale = np.linalg.norm(design, axis=0)
    coefficients, _, rank, singular = np.linalg.lstsq(design/scale, target, rcond=None)
    coefficients = coefficients/scale
    if rank != design.shape[1]:
        raise RuntimeError('rank-deficient surface-potential fit')
    pure = cheb2poly(coefficients[:degree_t+1])
    tensor = coefficients[degree_t+1:-1].reshape(degree_s+1, degree_t+1)
    # Convert fitted Chebyshev basis into explicit powers of sqrt(SA/SAmax)
    # and normalized temperature, for simple analytic Horner derivatives.
    power_x = np.zeros_like(tensor)
    for j in range(degree_t+1):
        polynomial = Chebyshev(tensor[:, j]).convert(kind=Polynomial)(Polynomial([-1., 2.]))
        power_x[:len(polynomial.coef), j] = polynomial.coef
    power = np.zeros_like(tensor)
    for i in range(degree_s+1):
        row = cheb2poly(power_x[i])
        power[i, :len(row)] = row
    return dict(format='surface-gibbs-reference-v1', gsw_version=gsw.__version__,
        numpy_version=np.__version__, reference='https://teos-10.github.io/GSW-Python/',
        training=dict(samples=len(target), degree_sqrt_salinity=degree_s,
                      degree_temperature=degree_t, scaled_condition_number=float(singular[0]/singular[-1]),
                      maximum_gibbs_residual_J_kg=float(np.abs(design@coefficients-target).max())),
        salinity_assumption='SA=SR=(35.16504/35)*SP; reference composition, no geographic anomaly',
        practical_salinity_range=[0., spmax], temperature_ITS90_range_C=[tmin, tmax],
        ups=ups, sa_scale=samax, temperature_midpoint=midpoint, temperature_halfspan=halfspan,
        pure_power_coefficients=pure.tolist(), salt_power_coefficients=power.tolist(),
        entropy_log_coefficient=float(coefficients[-1]),
        fitter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = fit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['training']))
