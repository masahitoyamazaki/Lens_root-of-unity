"""BKMS-type phase at the symmetric point: W(a,b) = omega^{s Q(a,b)} prod_i e^{s' i pi d_i^2/(2M)} F_theta(d_i), d=a-b."""
import numpy as np
from lensnp import PI
from trig2 import Simple

class BK(Simple):
    def __init__(self, n, M, s, sp):
        Simple.__init__(self, n, M, 0); self.s, self.sp = s, sp
    def logW(self, th, A, B):
        n, M = self.n, self.M
        Q = sum(A[i]*sum(A[j]-B[j] for j in range(i+1)) for i in range(n-1))
        return sum(self.logF(th, A[i]-B[i]) + self.sp*1j*PI*(A[i]-B[i])**2/(2*M) for i in range(n)) + self.s*2j*PI/M*Q

for (n, M) in [(3, 2), (3, 3), (3, 4), (3, 5), (4, 2), (4, 3)]:
    out = ["s=%+d,s'=%+d:%.0e" % (s, sp, BK(n, M, s, sp).test()[1]) for s in (1, -1) for sp in (1, -1)]
    print("n=%d M=%d " % (n, M), "  ".join(out))
