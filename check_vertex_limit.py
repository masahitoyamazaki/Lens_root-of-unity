from lensnp import *
from omega import logomega
def logpair(x, m, r, t1, t2):
    eta = -1j*PI*(t1+t2)/2   # exact identity verified in check_vertex_exact.py
    return eta*r/2 - 2*eta*m**2/r + logthbar(1, x - PI*m*t1, r*t1) + logthbar(1, x + PI*m*t2, r*t2)
def wrap(d): return d - 2j*PI*np.round(d.imag/(2*PI))
for (r,N) in [(2,3),(3,2),(1,4)]:
    tau = 0.07+0.3j; lam = PI*(1+r*N*tau)/(r*N)
    for xi in [0.21+0.04j, 0.9-0.03j]:
        for a in [1,2,5,-1]:
            vals=[]
            for h in [4e-3,2e-3,1e-3]:
                t2 = 1/(r*N)+1j*h/(PI*r)
                vals.append(logpair(xi+PI*a/(r*N), -a, r, tau, t2) - logpair(xi, 0, r, tau, t2))
            rich = 2*vals[2]-vals[1]
            k = np.floor((N*xi).real/PI)   # branch: reduce Re(N xi) into (0,pi)
            xr = xi - k*PI/N
            guess = 1j*lam*a**2/r + 1j*a*(2*N*xr-PI)/(r*N) + logthbar(1, xi+a*lam, r*tau) - logthbar(1, xi, r*tau)
            print("r,N=%d,%d xi=%s a=%d  |diff|=%.1e"%(r,N,xi,a,abs(wrap(rich-guess))))
# lemma for edges at larger Re xi
r,N=2,3; tau=0.07+0.3j
for xi in [1.5+0.05j, -0.8+0.02j]:
    for a in [1,3]:
        vals=[]
        for h in [4e-3,2e-3,1e-3]:
            t2 = 1/(r*N)+1j*h/(PI*r)
            vals.append(logPhi(xi+PI*a/(r*N), -a, r, tau, t2) - logPhi(xi, 0, r, tau, t2))
        print("edge lemma xi=%s a=%d |diff|=%.1e"%(xi,a,abs(wrap(2*vals[2]-vals[1]-logomega(xi,a,r,N,tau)))))
