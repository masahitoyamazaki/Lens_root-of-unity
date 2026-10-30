"""Closed form of the discrete lens gamma omega(xi|a):
 log omega = (a/(rN)) log thbar4(N xi|tauhat) + P(a,xi) - sum_{k=0}^{a-1} log thbar1(xi+(k+1/2)lam | r tau)
 P(a,xi) = E0(xi+pi a/(rN),-a) + i a xi + i lam a^2/2 + i a (pi r tau/4 - pi/2)
"""
import sympy as sp
from lensnp import *
from omega import logomega
a, r, N, tau, xi = sp.symbols('a r N tau xi')
lam = sp.pi*(1+r*N*tau)/(r*N)
m = -a; z = xi + sp.pi*a/(r*N); t2 = 1/(r*N)
E0 = sp.I*m*(r-m)/(6*r)*(6*z + sp.pi*(r-2*m)*(tau - t2 - 1))
P = E0 + sp.I*a*xi + sp.I*lam*a**2/2 + sp.I*a*(sp.pi*r*tau/4 - sp.pi/2)
P = sp.expand(P)
print("P(a,xi) =", sp.collect(P, [a**3, a**2, a], sp.factor))
Pf = sp.lambdify((a, r, N, tau, xi), P, 'numpy')
def logomega_cf(x, A, R, NN, t):
    L = PI*(1+R*NN*t)/(R*NN)
    s = A/(R*NN)*logthbar(4, NN*x, NN*t+1/R) + Pf(A, R, NN, t, x)
    if A >= 0:
        s -= sum(logthbar(1, x+(k+0.5)*L, R*t) for k in range(A))
    else:
        s += sum(logthbar(1, x+(k+0.5)*L, R*t) for k in range(A, 0))
    return s
def wrap(d): return d - 2j*PI*np.round(d.imag/(2*PI))
for (R,NN) in [(2,3),(3,2),(1,5)]:
    t = 0.07+0.3j; x = 0.21+0.04j
    print(R, NN, [float(abs(wrap(logomega_cf(x,A,R,NN,t)-logomega(x,A,R,NN,t)))) for A in [-3,-1,1,2,5,R*NN]])
