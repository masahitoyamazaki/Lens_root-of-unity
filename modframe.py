"""Discrete Z_M^{n-1} model in the modular frame (depends only on M=rN, T=r tau/(1+rN tau)).
  w(phi|a) = thbar2(phi|M T)^{a/M} / prod_{k=0}^{a-1} thbar1((phi+(k+1/2)pi)/M | T)
  W_theta(A,B) = prod_{i,j} w(phi_Ai - phi_Bj + theta | a_Ai - a_Bj)
  S(phi,a)     = prod_{i<j} thbar4((phi_ij + pi a_ij)/M + pi T/2 | T)
(gauge factors e^{p(a)} and a-independent factors dropped)."""
import itertools
import numpy as np
from lensnp import logthbar, PI


class ModModel:
    def __init__(self, n, M, T, eps=None):
        self.n, self.M, self.T = n, M, T
        # chamber signs eps[i][j] = sign Re(N xi_ij) (i<j) for the internal spin; default +1
        self.eps = eps

    def logw(self, phi, a):
        M, T = self.M, self.T
        s = a / M * logthbar(2, phi, M * T)
        if a >= 0:
            s -= sum(logthbar(1, (phi + (k + 0.5) * PI) / M, T) for k in range(a))
        else:
            s += sum(logthbar(1, (phi + (k + 0.5) * PI) / M, T) for k in range(a, 0))
        return s

    def logW(self, th, PA, AA, PB, AB):
        n = self.n
        return sum(self.logw(PA[i] - PB[j] + th, AA[i] - AB[j]) for i in range(n) for j in range(n))

    def logS(self, P, A, eps=None):
        n, M, T = self.n, self.M, self.T
        s = 0
        for i in range(n):
            for j in range(i + 1, n):
                e = 1 if eps is None else eps[i][j]
                aij = A[i] - A[j]
                s += logthbar(1, (P[i] - P[j] + PI * aij) / M, T) - 1j * PI * e * aij / M
        return s

    def sides(self, rap, P, A, Ph1, Ph2):
        """returns (LHS, RHS) of the discrete star-star relation; P = (PI,PJ,PK,PL), A likewise."""
        u, up, v, vp = rap
        hf = PI / 2
        (PI_, PJ, PK, PL), (AI, AJ, AK, AL) = P, A
        n, M = self.n, self.M
        L = R = 0
        for ah in itertools.product(range(M), repeat=n - 1):
            ah = list(ah) + [-sum(ah)]
            L += np.exp(self.logS(Ph1, ah, self.eps1) + self.logW(hf - (u - v), PK, AK, Ph1, ah)
                        + self.logW(hf - (up - vp), PJ, AJ, Ph1, ah)
                        + self.logW(up - v, Ph1, ah, PI_, AI) + self.logW(u - vp, Ph1, ah, PL, AL))
            R += np.exp(self.logS(Ph2, ah, self.eps2) + self.logW(hf - (u - v), Ph2, ah, PJ, AJ)
                        + self.logW(hf - (up - vp), Ph2, ah, PK, AK)
                        + self.logW(up - v, PL, AL, Ph2, ah) + self.logW(u - vp, PI_, AI, Ph2, ah))
        L *= np.exp(self.logW(vp - v, PL, AL, PK, AK) + self.logW(up - u, PL, AL, PJ, AJ))
        R *= np.exp(self.logW(vp - v, PJ, AJ, PI_, AI) + self.logW(up - u, PK, AK, PI_, AI))
        return L, R

    def test(self, rap, P, Ph1, Ph2, trials=5, seed=0, eps1=None, eps2=None):
        self.eps1, self.eps2 = eps1, eps2
        rng = np.random.default_rng(seed)
        n, M = self.n, self.M
        rat = []
        for t in range(trials):
            A = []
            for _ in range(4):
                a = list(rng.integers(0, M, n - 1)) if t else [0] * (n - 1)
                a.append(-sum(a)); A.append(a)
            L, R = self.sides(rap, P, A, Ph1, Ph2)
            rat.append(L / R)
        rat = np.array(rat)
        return rat[0], np.max(abs(rat / rat[0] - 1))


if __name__ == "__main__":
    # sanity check at a generic saddle point (taken from the xi-frame solver)
    from discrete_ssr import Model
    for (n, r, N) in [(2, 1, 3), (3, 1, 2), (3, 2, 2)]:
        tau = 0.05 + 0.35j
        X = Model(n, r, N, tau)
        lam = X.lam; M = r * N; T = r * tau / (1 + r * N * tau)
        rng = np.random.default_rng(0)
        def rxi():
            x = rng.uniform(-0.25, 0.25, n) * PI / N; return x - x.mean()
        XI, XJ, XK, XL = [rxi() for _ in range(4)]
        u, up, v, vp = rap = (0.9, 0.5, 0.2, 0.1); hf = PI / 2
        legs1 = [(hf - (u - v), XK, -1), (hf - (up - vp), XJ, -1), (up - v, XI, +1), (u - vp, XL, +1)]
        legs2 = [(hf - (u - v), XJ, +1), (hf - (up - vp), XK, +1), (up - v, XL, -1), (u - vp, XI, -1)]
        xh1, _ = X.saddle(legs1, np.linspace(-0.3, 0.3, n) * PI / N)
        xh2, _ = X.saddle(legs2, xh1)
        f = PI / lam
        mm = ModModel(n, M, T)
        sg = lambda x: [[np.sign((N * (x[i] - x[j])).real) for j in range(n)] for i in range(n)]
        R0, dev = mm.test(rap, [f * XI, f * XJ, f * XK, f * XL], f * xh1, f * xh2, eps1=sg(xh1), eps2=sg(xh2))
        print("generic saddle n,r,N=%d,%d,%d: max rel dev of LHS/RHS over ext. spins = %.1e" % (n, r, N, dev))
