from lensnp import *
r, N = 2, 3
tau = 0.05+0.55j
xi = 0.21+0.03j
def tau2(h): return 1/(r*N) + 1j*h/(PI*r)
tauhat = N*tau + 1/r
for h in [1e-2, 3e-3, 1e-3]:
    t2 = tau2(h)
    for a in [0, 1, 2]:
        z = xi + PI*a/(r*N) + PI*t2 - 1j*h/r   # = xi + pi(a+1)/(rN)
        D = logPhi(z, -a-1, r, tau, t2) - logPhi(xi + PI*a/(r*N)+PI*t2, -a-1, r, tau, t2)
        print(h, a, D)
print("cands: 1/(rN) log th4bar(N xi|tauhat) =", logthbar(4, N*xi, tauhat)/(r*N))
print("       1/(rN) log th3bar(N xi|tauhat) =", logthbar(3, N*xi, tauhat)/(r*N))
