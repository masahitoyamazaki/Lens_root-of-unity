"""Trigonometric limit in BKMS form:
  W_theta(a,b) = omega^{-Q(a,b)} prod_{i=1}^n prod_{k=0}^{a_i-b_i-1} 1/(1 - t omega^{-(k+1/2)}),  t = e^{-2 i theta/M},
  S = 1,  Q(v,w) = sum_{i=1}^{n-1} v_i sum_{j<=i}(v_j-w_j),  omega = e^{2 pi i/M}.
(and the complex-conjugate variant, with omega^{+Q})"""
import numpy as np
from lensnp import PI
from trig2 import Simple

class BKform(Simple):
    def __init__(self, n, M, conj=False):
        Simple.__init__(self, n, M, 0); self.conj = conj
    def logfac(self, th, k):
        M = self.M
        if self.conj:
            return np.log(1 - np.exp(2j*th/M)*np.exp(2j*PI*(k+0.5)/M))
        return np.log(1 - np.exp(-2j*th/M)*np.exp(-2j*PI*(k+0.5)/M))
    def logW(self, th, A, B):
        n, M = self.n, self.M
        Q = sum(A[i]*sum(A[j]-B[j] for j in range(i+1)) for i in range(n-1))
        s = (1 if self.conj else -1)*2j*PI/M*Q
        for i in range(n):
            d = A[i]-B[i]
            s += -sum(self.logfac(th, k) for k in range(d)) if d >= 0 else sum(self.logfac(th, k) for k in range(d, 0))
        return s

for (n, M) in [(2, 3), (2, 5), (3, 3), (3, 4), (3, 5), (4, 3), (5, 2), (5, 3)]:
    print("n=%d M=%d  BKMS form: %.1e   conjugate: %.1e" % (n, M, BKform(n, M).test()[1], BKform(n, M, True).test()[1]))
