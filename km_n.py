"""Z_n-symmetric ('sl(n) Kashiwara-Miwa') point: all spins at phi0_p=(p-(n+1)/2) w0, w0=pi(MT-nu)/n, nu in {0,1}.
Test: the fractional powers thbar2^{a/M} can be dropped, leaving pure theta products:
  W_theta(a,b) = prod_{i,j} 1/prod_{k=0}^{a_i-b_j-1} thbar1( ((i-j) w0 + theta + (k+1/2) pi)/M | T )
  S(a)         = prod_{i<j} thbar1( (pi a_ij - (j-i) w0)/M | T ) e^{-i pi a_ij/M} * exp(-2 pi i <rho,a>/M)."""
import itertools, sys
import numpy as np
from lensnp import logthbar, PI
from modframe import ModModel

class KMn(ModModel):
    def logS(self, P, A, eps=None):
        # vertex correction from the fractional powers: exp(-2 pi i <rho,a>/M), rho=(0,1,...,n-1)
        return ModModel.logS(self, P, A, eps) - 2j * PI * sum(i * A[i] for i in range(self.n)) / self.M
    def logw(self, phi, a):
        M, T = self.M, self.T
        if a >= 0:
            return -sum(logthbar(1, (phi + (k + 0.5) * PI) / M, T) for k in range(a))
        return sum(logthbar(1, (phi + (k + 0.5) * PI) / M, T) for k in range(a, 0))

for (n, M) in [(2, 3), (2, 4), (3, 2), (3, 3), (4, 2)]:
    for nu in [0, 1]:
        T = 0.1 + 0.25j
        w0 = PI * (M * T - nu) / n
        P0 = np.array([(p - (n + 1) / 2) * w0 for p in range(1, n + 1)])
        eps = [[1] * n for _ in range(n)]
        R0, dev = KMn(n, M, T).test((0.9, 0.5, 0.2, 0.1), [P0] * 4, P0, P0, trials=4, eps1=eps, eps2=eps)
        print("n=%d M=%d nu=%d : pure-theta weights, dev=%.1e, R=%s" % (n, M, nu, dev, np.round(R0, 8)))
