"""Simplest form of the trigonometric model:
  W_theta(a,b) = prod_i F_theta(a_i-b_i) * exp(-(i pi/M) sum_{i>j}(a_i b_j - a_j b_i)),  S = 1,
  F_theta(d) = prod_{k=0}^{d-1} 1/sin((theta+(k+1/2)pi)/M).
Variants: without chiral phase; with chiral phase sign flipped."""
import numpy as np
from lensnp import PI
from trig import Trig

class Simple(Trig):
    def __init__(self, n, M, chiral=-1):
        Trig.__init__(self, n, M); self.chiral = chiral
    def logF(self, th, d):
        if d >= 0:
            return -sum(np.log(complex(np.sin((th+(k+0.5)*PI)/self.M))) for k in range(d))
        return sum(np.log(complex(np.sin((th+(k+0.5)*PI)/self.M))) for k in range(d, 0))
    def logW(self, th, A, B):
        n, M = self.n, self.M
        return (sum(self.logF(th, A[i]-B[i]) for i in range(n))
                + self.chiral*1j*PI/M*sum(A[i]*B[j]-A[j]*B[i] for i in range(n) for j in range(i)))
    def logS(self, A):
        return 0

for (n, M) in [(2, 3), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3)]:
    res = {c: Simple(n, M, c).test()[1] for c in (-1, 0, +1)}
    print("n=%d M=%d  chiral=-1: %.1e   chiral=0: %.1e   chiral=+1: %.1e" % (n, M, res[-1], res[0], res[1]))
