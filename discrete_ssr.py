"""End-to-end test of the discrete (Z_{rN})^{n-1} star-star relation obtained from the
root-of-unity limit of the A_{n-1} lens-elliptic-gamma star-star relation.

Parametrisation at the root of unity  q = e^{i pi/(rN)} e^{-hbar/r}:
  x_{c,i} = xi_{c,i} + pi a_{c,i}/(rN),   m_{c,i} = -a_{c,i} mod r,   sum_i xi_{c,i}=0, sum_i a_{c,i}=0
  lambda  = pi (1 + rN tau)/(rN),   i*alpha = theta*lambda/pi,   i*eta -> lambda/2.

O(1) discrete weights (a-independent factors dropped):
  edge   : W_theta(A,B)  = prod_{i,j} omega(xi_{A,i}-xi_{B,j} + theta lambda/pi | a_{A,i}-a_{B,j})
  vertex : S(xi,a)       = prod_{i<j} exp(i lam a_ij^2/r + i a_ij (2N xi~_ij - pi)/(rN)) thbar1(xi_ij + a_ij lam | r tau)
Saddle point: leading action, d/dz f(z) = (i/N) log thbar4(N z | N tau + 1/r) for edges,
  vertex pair leading derivative s(xi) = -(2N xi~ - pi)/N  (xi~ reduced to 0<Re N xi~<pi).
"""
import itertools
import numpy as np
from lensnp import logthbar, PI
from omega import T


class Model:
    def __init__(self, n, r, N, tau):
        self.n, self.r, self.N, self.tau = n, r, N, tau
        self.rN = r * N
        self.lam = PI * (1 + r * N * tau) / (r * N)
        self.tauhat = N * tau + 1.0 / r

    # ---------- O(1) discrete weights ----------
    def logomega(self, z, a):
        r, N = self.r, self.N
        L4 = logthbar(4, N * z, self.tauhat) / (r * N)
        if a >= 0:
            return a * L4 + sum(T(z + PI * k / (r * N), k, r, N, self.tau) for k in range(a))
        return a * L4 - sum(T(z + PI * k / (r * N), k, r, N, self.tau) for k in range(a, 0))

    def logW(self, theta, xA, aA, xB, aB):
        c = theta * self.lam / PI
        return sum(self.logomega(xA[i] - xB[j] + c, aA[i] - aB[j])
                   for i in range(self.n) for j in range(self.n))

    def logS(self, x, a):
        r, N, lam = self.r, self.N, self.lam
        s = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                xij, aij = x[i] - x[j], a[i] - a[j]
                k = np.floor((N * xij).real / PI)
                xr = xij - k * PI / N
                s += (1j * lam * aij ** 2 / r + 1j * aij * (2 * N * xr - PI) / (r * N)
                      + logthbar(1, xij + aij * lam, r * self.tau))
        return s

    # ---------- leading order: saddle point ----------
    def dlead_edge(self, z):
        return 1j / self.N * logthbar(4, self.N * z, self.tauhat)

    def dlead_S(self, xij):
        N = self.N
        k = np.floor((N * xij).real / PI)
        xr = xij - k * PI / N
        return -(2 * N * xr - PI) / N

    def grad(self, xh, legs):
        """legs: list of (theta, X, sign) ; sign=+1 if xh is first argument of W, -1 if second."""
        n = self.n
        g = np.zeros(n, dtype=complex)
        for k in range(n):
            for theta, X, sgn in legs:
                c = theta * self.lam / PI
                for i in range(n):
                    if sgn > 0:
                        g[k] += self.dlead_edge(xh[k] - X[i] + c)
                    else:
                        g[k] -= self.dlead_edge(X[i] - xh[k] + c)
            for j in range(n):
                if j != k:
                    g[k] += self.dlead_S(xh[k] - xh[j])
        return g

    def saddle(self, legs, x0, tol=1e-13, maxit=100):
        n = self.n
        y = np.concatenate([np.asarray(x0[:n - 1], dtype=complex), [0j]])

        def F(y):
            xh = np.concatenate([y[:n - 1], [-np.sum(y[:n - 1])]])
            return self.grad(xh, legs) - y[n - 1]
        for it in range(maxit):
            f = F(y)
            if np.max(abs(f)) < tol:
                break
            J = np.zeros((n, n), dtype=complex)
            h = 1e-7
            for c in range(n):
                dy = np.zeros(n, dtype=complex); dy[c] = h
                J[:, c] = (F(y + dy) - f) / h
            step = np.linalg.solve(J, f)
            t = 1.0
            while t > 1e-4:
                yn = y - t * step
                if np.max(abs(F(yn))) < np.max(abs(f)):
                    break
                t /= 2
            y = yn
        xh = np.concatenate([y[:n - 1], [-np.sum(y[:n - 1])]])
        return xh, np.max(abs(F(y)))


def run(n, r, N, tau, rap, seed=0, trials=4, verbose=True, return_data=False):
    M = Model(n, r, N, tau)
    rng = np.random.default_rng(seed)
    u, up, v, vp = rap
    hf = PI / 2

    def rxi():
        x = rng.uniform(-0.25, 0.25, n) * PI / N
        return x - x.mean()
    XI, XJ, XK, XL = [rxi() for _ in range(4)]
    # star 1: Wbar_{u-v}(K,h) Wbar_{u'-v'}(J,h) W_{u'-v}(h,I) W_{u-v'}(h,L)
    legs1 = [(hf - (u - v), XK, -1), (hf - (up - vp), XJ, -1), (up - v, XI, +1), (u - vp, XL, +1)]
    # star 2: Wbar_{u-v}(h,J) Wbar_{u'-v'}(h,K) W_{u'-v}(L,h) W_{u-v'}(I,h)
    legs2 = [(hf - (u - v), XJ, +1), (hf - (up - vp), XK, +1), (up - v, XL, -1), (u - vp, XI, -1)]
    x0 = np.linspace(-0.3, 0.3, n) * PI / N
    xh1, e1 = M.saddle(legs1, x0)
    xh2, e2 = M.saddle(legs2, xh1)
    if verbose:
        print("saddle 1:", np.round(xh1, 6), "res", e1)
        print("saddle 2:", np.round(xh2, 6), "res", e2)
    rN = M.rN

    def rand_a():
        a = list(rng.integers(0, rN, n - 1)); a.append(-sum(a)); return a

    def star_sum(xh, legs_a):
        tot = 0
        for ah in itertools.product(range(rN), repeat=n - 1):
            ah = list(ah) + [-sum(ah)]
            s = M.logS(xh, ah)
            for theta, X, A, sgn in legs_a:
                s += M.logW(theta, xh, ah, X, A) if sgn > 0 else M.logW(theta, X, A, xh, ah)
            tot += np.exp(s)
        return tot

    ratios = []
    for t in range(trials):
        AI, AJ, AK, AL = (rand_a() for _ in range(4)) if t > 0 else ([0] * n,) * 4
        F1 = np.exp(M.logW(vp - v, XL, AL, XK, AK) + M.logW(up - u, XL, AL, XJ, AJ)) * star_sum(
            xh1, [(hf - (u - v), XK, AK, -1), (hf - (up - vp), XJ, AJ, -1), (up - v, XI, AI, +1), (u - vp, XL, AL, +1)])
        F2 = np.exp(M.logW(vp - v, XJ, AJ, XI, AI) + M.logW(up - u, XK, AK, XI, AI)) * star_sum(
            xh2, [(hf - (u - v), XJ, AJ, +1), (hf - (up - vp), XK, AK, +1), (up - v, XL, AL, -1), (u - vp, XI, AI, -1)])
        ratios.append(F1 / F2)
        if verbose:
            print("  ext a:", AI, AJ, AK, AL, " F1/F2 =", F1 / F2)
    ratios = np.array(ratios)
    if return_data:
        return ratios[0], (XI, XJ, XK, XL, xh1, xh2)
    return np.max(abs(ratios / ratios[0] - 1))


if __name__ == "__main__":
    import sys
    n, r, N = (int(s) for s in sys.argv[1:4])
    dev = run(n, r, N, tau=0.05 + 0.35j, rap=(0.9, 0.5, 0.2, 0.1))
    print("max relative deviation of F1/F2 across external spins:", dev)
