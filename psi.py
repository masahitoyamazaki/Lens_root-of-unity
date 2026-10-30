"""O(1) part Psi(z) of log Phi(z,0) at the root of unity:
   log Phi(z,0) = F(z)/hbar + Psi(z) + O(hbar),  F'(z) = (i/N) log thbar4(N z|tauhat), F(0)=0."""
import numpy as np
from scipy.integrate import quad
from lensnp import logPhi, logthbar, PI

def F(z, r, N, tau):
    tauhat = N*tau + 1/r
    f = lambda t: (1j/N*logthbar(4, N*z*t, tauhat)*z)
    re = quad(lambda t: f(t).real, 0, 1, epsabs=1e-14, epsrel=1e-14, limit=200)[0]
    im = quad(lambda t: f(t).imag, 0, 1, epsabs=1e-14, epsrel=1e-14, limit=200)[0]
    return re + 1j*im

def Psi_num(z, r, N, tau, hs=(2e-3, 1e-3)):
    Fz = F(z, r, N, tau)
    v = [logPhi(z, 0, r, tau, 1/(r*N)+1j*h/(PI*r)) - Fz/h for h in hs]
    return 2*v[1]-v[0]

def Psi_1709(z, r, N, tau):
    """1709 eq.(3.4) (slead) at m=0"""
    tauhat = N*tau + 1/r
    s = 0
    for k1 in range(N):
        for k2 in range(r):
            e = (r*N - 2*r*(k1+1) + 2*k2 + 1)/(2*r*N)
            s += e*logthbar(4, z + PI/N*(k1+1) - PI*tauhat/(2*N)*(2*k2+1) + PI*r*tau/2, r*tau)
    return s

def Psi_torsion(z, r, N, tau):
    """conjectured simplification: prod_{j=1}^{rN} thbar1(z + (j-1/2) lam | r tau)^{(rN-2j+1)/(2rN)} (up to exp-linear)"""
    lam = PI*(1+r*N*tau)/(r*N)
    return sum((r*N-2*j+1)/(2*r*N)*logthbar(1, z+(j-0.5)*lam, r*tau) for j in range(1, r*N+1))

if __name__ == "__main__":
    for (r, N) in [(1, 1), (1, 3), (2, 2), (3, 2)]:
        tau = 0.05+0.4j
        for z in [0.13+0.03j, -0.21+0.05j, 0.3-0.02j]:
            pn = Psi_num(z, r, N, tau); p9 = Psi_1709(z, r, N, tau); pt = Psi_torsion(z, r, N, tau)
            print("r,N=%d,%d z=%s  num=%s  1709=%s  torsion=%s" % (r, N, z, np.round(pn, 6), np.round(p9, 6), np.round(pt, 6)))
