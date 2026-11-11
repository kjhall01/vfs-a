"""Ocean-owned discrete surface Gibbs means for the fitted GSW reference.

Symmetric coordinate-increment divided differences give
Delta b + eta_face*Delta T + S_face*Delta mu = 0, b=G-S*mu.
No velocity, atmospheric imports, finite differences or forward autograd.
The underlying thermodynamics remain derived from GSW, not a new EOS.
"""
import torch


def polynomial_secant(a, b, coefficients):
    """(P(b)-P(a))/(b-a), including the exact polynomial equal-input limit."""
    value = torch.zeros_like(a)+coefficients[-1]
    secant = torch.zeros_like(a)
    for c in reversed(coefficients[:-1]):
        secant = value+b*secant
        value = a*value+c
    return secant


def power_salinity_secant(x, y, n):
    """Delta sqrt(SA)^n / Delta SA, using polynomial factorization."""
    if n == 0:
        return torch.zeros_like(x)
    series = torch.ones_like(x)
    power = torch.ones_like(x)
    for _ in range(n-1):
        power = power*x
        series = series*y+power
    return series/(x+y)


def log_secant(a, b):
    """Delta log(SA)/Delta SA, with a stable equal-state limit."""
    x = (b-a)/(b+a)
    close = x.abs() < 1e-3
    series = 1+x.square()/3+x.pow(4)/5+x.pow(6)/7
    # log differences remain finite for large ratios where atanh(x) rounds
    # to infinity. Mask the unused denominator before division for backward.
    den = torch.where(close, torch.ones_like(a), b-a)
    exact = (torch.log(b)-torch.log(a))/den
    return torch.where(close, series/(.5*(a+b)), exact)


def gibbs_face(reference, si, ti, sj, tj):
    """Input SP and explicit-model-scale theta; return eta*, SP* and audit.

    D_T and D_S are symmetric coordinate-increment discrete derivatives.
    dS=(mean(mu)-D_S G)/D_S mu; S*=mean(S)+dS;
    eta*=-D_T G-dS*D_T mu. D_S mu>0 is checked, not assumed.
    Positive salinity is required. These are thermodynamic means, not bounds.
    """
    vi = reference.evaluate(si, ti, entropy_derivatives=True)
    vj = reference.evaluate(sj, tj, entropy_derivatives=True)
    d = reference.data
    ups, span = d['ups'], d['temperature_halfspan']
    sai, saj = ups*si, ups*sj
    xi, xj = sai.sqrt(), saj.sqrt()
    yi = (reference.to_90*ti-d['temperature_midpoint'])/span
    yj = (reference.to_90*tj-d['temperature_midpoint'])/span
    # Evaluate row polynomials at endpoint T using a secant with P(0) removed.
    def value(y, row):
        out = torch.zeros_like(y)+row[-1]
        for c in reversed(row[:-1]):
            out = out*y+c
        return out
    dtg = polynomial_secant(yi, yj, d['pure_power_coefficients'])/span
    dsg, dsmu, dtmu = [torch.zeros_like(dtg) for _ in range(3)]
    for i, row in enumerate(d['salt_power_coefficients']):
        denominator = d['sa_scale']**(.5*i)
        cmean = .5*(value(yi, row)+value(yj, row))
        dc = polynomial_secant(yi, yj, row)/span
        dtg = dtg+.5*(xi.pow(i+2)+xj.pow(i+2))*dc/denominator
        dsg = dsg+ups*power_salinity_secant(xi, xj, i+2)*cmean/denominator
        dsmu = dsmu+ups**2*(1+.5*i)*power_salinity_secant(xi, xj, i)*cmean/denominator
        dtmu = dtmu+ups*(1+.5*i)*.5*(xi.pow(i)+xj.pow(i))*dc/denominator
    li, lj = torch.log(sai/35.16504), torch.log(saj/35.16504)
    dl = log_secant(sai, saj)
    coefficient = d['entropy_log_coefficient']
    tmean = .5*(vi['kelvin']+vj['kelvin'])
    dtg = dtg+coefficient*.5*(sai*li+saj*lj)
    dsg = dsg+ups*coefficient*tmean*(.5*(li+lj)+.5*(sai+saj)*dl)
    dsmu = dsmu+ups**2*coefficient*tmean*dl
    dtmu = dtmu+ups*coefficient*(.5*(li+lj)+1)
    mui = vi['h_s']-vi['kelvin']*vi['entropy_s']
    muj = vj['h_s']-vj['kelvin']*vj['entropy_s']
    if not torch.isfinite(dsmu).all() or (dsmu <= 0).any():
        raise FloatingPointError('surface chemical susceptibility must be finite and positive')
    shift = (.5*(mui+muj)-dsg)/dsmu
    salt = .5*(si+sj)+shift
    entropy = -dtg-shift*dtmu
    bi, bj = vi['gibbs']-si*mui, vj['gibbs']-sj*muj
    terms = (bj-bi, entropy*(vj['kelvin']-vi['kelvin']), salt*(muj-mui))
    return entropy, salt, dict(identity=sum(terms), scale=sum(x.abs() for x in terms),
                              chemical_susceptibility=dsmu,
                              composition_entropy=.5*(vi['entropy_s']+vj['entropy_s']))


def temperature_from_entropy(reference, sp, entropy):
    """Explicit analytic bracketed Newton inversion; no clipping or forward AD."""
    lo, hi = [torch.full_like(entropy, t/reference.to_90) for t in reference.data['temperature_ITS90_range_C']]
    el, eh = reference.evaluate(sp, lo)['entropy'], reference.evaluate(sp, hi)['entropy']
    if not torch.isfinite(entropy).all() or ((entropy < el)|(entropy > eh)).any():
        raise ValueError('entropy outside the surface-reference domain')
    theta = lo+(hi-lo)*(entropy-el)/(eh-el)
    for _ in range(12):
        values = reference.evaluate(sp, theta)
        error = values['entropy']-entropy
        lo = torch.where(error < 0, theta, lo)
        hi = torch.where(error > 0, theta, hi)
        proposal = theta-error/values['entropy_theta']
        theta = torch.where((proposal >= lo)&(proposal <= hi), proposal, .5*(lo+hi))
    residual = (reference.evaluate(sp, theta)['entropy']-entropy).abs()
    if (residual > 128*torch.finfo(entropy.dtype).eps*(entropy.abs()+20)).any():
        raise FloatingPointError('surface entropy inversion failed its residual check')
    return theta
