from lensnp import *
def T(z, a, r, N, tau):
    """exact shift factor at root of unity: Phi(z+pi tau2, m-1)/Phi(z,m), m=-a, tau2=1/(rN)"""
    t2 = 1/(r*N); m = -a
    p = np.exp(1j*PI*tau); zeta = np.exp(1j*PI/(r*N))
    dE = E(z+PI*t2, m-1, r, tau, t2) - E(z, m, r, tau, t2)
    y = np.exp(2j*z)*p**(2*r-2*m+1)*zeta
    return dE - np.log(qpoch(y, p**(2*r))*qpoch(p**(2*r)/y, p**(2*r)))
def logomega(xi, a, r, N, tau):
    tauhat = N*tau + 1/r
    s = a/(r*N)*logthbar(4, N*xi, tauhat)
    for k in range(a):
        s += T(xi + PI*k/(r*N), k, r, N, tau)
    for k in range(a, 0):
        s -= T(xi + PI*k/(r*N), k, r, N, tau)
    return s
if __name__=='__main__':
  for (r, N) in [(2,3),(3,2),(1,3),(3,1)]:
      tau = 0.07+0.3j; xi = 0.21+0.04j
      print("r,N=",r,N)
      for a in [1, 2, 4, r*N, r*N+1]:
          vals = []
          for h in [4e-3, 2e-3, 1e-3]:
              t2 = 1/(r*N) + 1j*h/(PI*r)
              vals.append(logPhi(xi+PI*a/(r*N), -a, r, tau, t2) - logPhi(xi, 0, r, tau, t2))
          # vals contain 1/h-divergent? no: leading parts cancel only if f is pi/N periodic... check
          rich = 2*vals[2]-vals[1]
          ex = logomega(xi, a, r, N, tau)
          d = rich-ex; d = d - 2j*PI*np.round(d.imag/(2*PI))
          print("  a=%d  numeric=%s  formula=%s  diff=%.2e"%(a, np.round(rich,6), np.round(ex,6), abs(d)))
