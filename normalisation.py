"""Prediction of the normalisation R in the discrete star-star relation and comparison with F1/F2.
R = exp[(K_R-K_L)_{(r,N,tau)} - (K_R-K_L)_{(1,1,tau1)}],  tau1 = N tau + 1/r - 1, spins xi -> N xi in (1,1).
K = sum of a-independent O(1) terms:
  W_theta(A,B) edge     : sum_ij Psi(z),  z = xi_Ai - xi_Bj + theta lam/pi
  Wbar_theta(A,B) edge  : sum_ij [Psi(z0) - log thbar4(N z0|tauhat)/(2rN)],  z0 = xi_Ai - xi_Bj + (pi/2-theta) lam/pi
  vertex (only in (1,1)): sum_{i<j} log thbar1(xi_ij | tau1)    [in (r,N) it is inside the discrete S]
"""
import numpy as np
from lensnp import logthbar, PI
from psi import Psi_1709
from discrete_ssr import Model, run
import discrete_ssr

def Kside(r, N, tau, legs_ext, legs_star, xh, include_vertex):
    """legs: list of (kind, theta, XA, XB) with kind 'W' or 'Wb'"""
    lam = PI*(1+r*N*tau)/(r*N); tauhat = N*tau + 1/r
    tot = 0
    for kind, th, XA, XB in legs_ext + legs_star:
        for xa in XA:
            for xb in XB:
                if kind == 'W':
                    z = xa - xb + th*lam/PI
                    tot += Psi_1709(z, r, N, tau)
                else:
                    z0 = xa - xb + (PI/2 - th)*lam/PI
                    tot += Psi_1709(z0, r, N, tau) - logthbar(4, N*z0, tauhat)/(2*r*N)
    if include_vertex:
        n = len(xh)
        for i in range(n):
            for j in range(i+1, n):
                tot += logthbar(1, xh[i]-xh[j], r*tau)
    return tot

def predict(n, r, N, tau, rap, XI, XJ, XK, XL, xh1, xh2):
    u, up, v, vp = rap
    def sides(r_, N_, tau_, s, inc):
        X = {k: s*np.asarray(val) for k, val in dict(I=XI, J=XJ, K=XK, L=XL, h1=xh1, h2=xh2).items()}
        KL = Kside(r_, N_, tau_, [('W', vp-v, X['L'], X['K']), ('W', up-u, X['L'], X['J'])],
                   [('Wb', u-v, X['K'], X['h1']), ('Wb', up-vp, X['J'], X['h1']),
                    ('W', up-v, X['h1'], X['I']), ('W', u-vp, X['h1'], X['L'])], X['h1'], inc)
        KR = Kside(r_, N_, tau_, [('W', vp-v, X['J'], X['I']), ('W', up-u, X['K'], X['I'])],
                   [('Wb', u-v, X['h2'], X['J']), ('Wb', up-vp, X['h2'], X['K']),
                    ('W', up-v, X['L'], X['h2']), ('W', u-vp, X['I'], X['h2'])], X['h2'], inc)
        return KR - KL
    tau1 = N*tau + 1/r - 1
    return np.exp(sides(r, N, tau, 1, False) - sides(1, 1, tau1, N, True))

if __name__ == "__main__":
    import sys
    n, r, N = (int(s) for s in sys.argv[1:4])
    tau = 0.05+0.35j; rap = (0.9, 0.5, 0.2, 0.1)
    out = run(n, r, N, tau, rap, trials=2, verbose=False, return_data=True)
    ratio, (XI, XJ, XK, XL, xh1, xh2) = out
    Rp = predict(n, r, N, tau, rap, XI, XJ, XK, XL, xh1, xh2)
    print("n,r,N=%d,%d,%d  F1/F2 = %s   R_pred = %s   ratio = %s" % (n, r, N, ratio, Rp, ratio/Rp))


def calK(r, N, tau, legs, xh):
    """simplified per-side functional:
    sum_W Psi(z) + sum_Wbar [Psi(z0) + (rN-1)/(2rN) log thbar4(N z0|tauhat)] - sum_{i<j} log thbar1(N xi_h,ij | tauhat)"""
    lam = PI*(1+r*N*tau)/(r*N); tauhat = N*tau + 1/r
    tot = 0
    for kind, th, XA, XB in legs:
        for xa in XA:
            for xb in XB:
                if kind == 'W':
                    tot += Psi_1709(xa-xb+th*lam/PI, r, N, tau)
                else:
                    z0 = xa-xb+(PI/2-th)*lam/PI
                    tot += Psi_1709(z0, r, N, tau) + (r*N-1)/(2*r*N)*logthbar(4, N*z0, tauhat)
    n = len(xh)
    for i in range(n):
        for j in range(i+1, n):
            tot -= logthbar(1, N*(xh[i]-xh[j]), tauhat)
    return tot

def predict_simplified(n, r, N, tau, rap, XI, XJ, XK, XL, xh1, xh2):
    u, up, v, vp = rap
    L = calK(r, N, tau, [('W', vp-v, XL, XK), ('W', up-u, XL, XJ), ('Wb', u-v, XK, xh1), ('Wb', up-vp, XJ, xh1),
                         ('W', up-v, xh1, XI), ('W', u-vp, xh1, XL)], xh1)
    R = calK(r, N, tau, [('W', vp-v, XJ, XI), ('W', up-u, XK, XI), ('Wb', u-v, xh2, XJ), ('Wb', up-vp, xh2, XK),
                         ('W', up-v, XL, xh2), ('W', u-vp, XI, xh2)], xh2)
    return np.exp(R - L)
