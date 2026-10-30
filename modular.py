"""Modular-frame (1709-type) form of the discrete weights.
tau' = rN tau/(1+rN tau),  phi = xi*pi/lam  (lam = pi(1+rN tau)/(rN)),  theta = c*pi/lam.
Claim: omega(xi|a) = exp(p(a)) * frak_w(phi|a) with
  frak_w(phi|a) = thbar2(phi | r tau')^{a/(rN)} / prod_{k=0}^{a-1} thbar1((phi+(k+1/2)pi)/(rN) | tau'/N)
and p(a) independent of xi.
"""
import numpy as np
from lensnp import logthbar, PI
from closed_form import logomega_cf, wrap

def logfw(phi, a, r, N, tp):
    s = a/(r*N)*logthbar(2, phi, r*tp)
    if a >= 0:
        s -= sum(logthbar(1, (phi+(k+0.5)*PI)/(r*N), tp/N) for k in range(a))
    else:
        s += sum(logthbar(1, (phi+(k+0.5)*PI)/(r*N), tp/N) for k in range(a, 0))
    return s

if __name__ == "__main__":
    for (r, N) in [(2, 3), (3, 2), (1, 4)]:
        tau = 0.07+0.3j
        lam = PI*(1+r*N*tau)/(r*N); tp = r*N*tau/(1+r*N*tau)
        print("r,N =", r, N)
        for a in [-2, -1, 1, 2, 3, 5]:
            ds = []
            for xi in [0.21+0.04j, -0.13+0.02j, 0.05-0.06j]:
                ds.append(logomega_cf(xi, a, r, N, tau) - logfw(xi*PI/lam, a, r, N, tp))
            ds = np.array(ds)
            spread = max(abs(wrap(ds - ds[0])))
            pred = 1j*PI*a**2/2 + 1j*PI*(r*N-1)*a**3/(3*r*r*N)
            print("  a=%2d  xi-independence %.1e   p(a)-[i pi a^2/2 + i pi (rN-1)a^3/(3r^2N)] = %s"
                  % (a, spread, np.round(wrap(ds[0]-pred), 6)))
