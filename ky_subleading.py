"""1709 eqs (3.3)-(3.6): the subleading term Phi^(0)(z,m) for all integers m (KYsub). Usage: python3 ky_subleading.py"""
import os, sys; sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from lensnp import logPhi, logthbar, PI
from psi import F
def logPhi0_1709(z, m, r, N, tau):
    th = N*tau+1/r; zh = z+PI*m/(r*N)
    s = 1j*m*(r-m)/(6*r*r*N)*(6*r*N*z+PI*(r-2*m)*(r*N*(tau-1)-1))
    for k1 in range(N):
        for k2 in range(r):
            e = (r*N-2*r*(k1+1)+2*k2-2*m+1)/(2*r*N)
            s += e*logthbar(4, zh+PI/N*(k1+1)-PI*th/(2*N)*(2*k2+1)+PI*r*tau/2, r*tau)
    sg = 1 if m >= 0 else -1
    for k in range(abs(m)):
        s += sg*logthbar(4, zh-sg*PI*th/(2*N)*(2*k+1)+PI*r*tau/2, r*tau)
    return s
def num(z, m, r, N, tau, hs=(2e-3,1e-3)):
    Fz = F(z+PI*m/(r*N), r, N, tau)
    v=[logPhi(z, m, r, tau, 1/(r*N)+1j*h/(PI*r)) - Fz/h for h in hs]
    return 2*v[1]-v[0]
tau=0.05+0.4j
for (r,N) in [(2,3),(3,2),(1,3)]:
    for m in range(-3, r+3):
        for xi in [0.11+0.02j, -0.23+0.05j]:
          z = xi - PI*m/(r*N)
          d = num(z,m,r,N,tau)-logPhi0_1709(z,m,r,N,tau)
          e=d-2j*PI*np.round((d/(2j*PI)).real); print(r,N,m,"%.1e"%abs(e))
