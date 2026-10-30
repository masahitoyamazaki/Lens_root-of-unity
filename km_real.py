"""Kashiwara-Miwa type points in the original variables xi (draft Sec. 5.3, analytic explanation of (w0)).
All spins at the real Z_n-symmetric configuration  xi_p = -rho_p * k pi/(nN)   (rho_p = p-(n+1)/2),
i.e. an n-torsion point of R/(pi/N)Z.  Checks, with the xi-frame weights of discrete_ssr.Model (canonical branch of
log thbar4(N z|tauhat), analytic in the strip |Im z| < Im(lambda)/2 which contains all edge arguments):
 (a) the configuration solves the saddle point equations of both stars (gradient independent of the component);
 (b) the fractional powers cancel identically: sum_{i,j} (a_i-b_j) log thbar4(N z_ij|tauhat) = 0 for sum a = sum b = 0;
 (c) the discrete star-star relation (dssr) holds, for gcd(k,n)=1, independently of Re tau;
 (d) for gcd(k,n)>1 two components coincide mod pi/N (Weyl wall) and the relation fails."""
import itertools, sys
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from lensnp import logthbar, PI
from discrete_ssr import Model


def check(n, r, N, tau, k, rap=(0.9, 0.5, 0.2, 0.1), trials=3, seed=0):
    X = Model(n, r, N, tau); hf = PI / 2; u, up, v, vp = rap
    x0 = np.array([-(p - (n - 1) / 2) * k * PI / (n * N) for p in range(n)], dtype=complex)
    legs1 = [(hf - (u - v), x0, -1), (hf - (up - vp), x0, -1), (up - v, x0, +1), (u - vp, x0, +1)]
    legs2 = [(hf - (u - v), x0, +1), (hf - (up - vp), x0, +1), (up - v, x0, -1), (u - vp, x0, -1)]
    g1, g2 = X.grad(x0, legs1), X.grad(x0, legs2)
    sad = max(np.ptp(g1.real) + np.ptp(g1.imag), np.ptp(g2.real) + np.ptp(g2.imag))
    rng = np.random.default_rng(seed); rN = r * N
    def ra():
        a = list(rng.integers(0, rN, n - 1)); a.append(-sum(a)); return a
    # (b) fractional powers
    fr = 0
    for th in [0.3, hf - 0.3]:
        c = th * X.lam / PI
        A, B = ra(), ra()
        s = sum((A[i] - B[j]) * logthbar(4, N * (x0[i] - x0[j] + c), X.tauhat) for i in range(n) for j in range(n))
        fr = max(fr, abs(s))
    # (c) star-star
    def star(legs_a):
        tot = 0
        for ah in itertools.product(range(rN), repeat=n - 1):
            ah = list(ah) + [-sum(ah)]
            s = X.logS(x0, ah)
            for th, A, sg in legs_a:
                s += X.logW(th, x0, ah, x0, A) if sg > 0 else X.logW(th, x0, A, x0, ah)
            tot += np.exp(s)
        return tot
    rat = []
    for t in range(trials):
        AI, AJ, AK, AL = (ra() for _ in range(4)) if t else ([0] * n,) * 4
        F1 = np.exp(X.logW(vp - v, x0, AL, x0, AK) + X.logW(up - u, x0, AL, x0, AJ)) * star(
            [(hf - (u - v), AK, -1), (hf - (up - vp), AJ, -1), (up - v, AI, +1), (u - vp, AL, +1)])
        F2 = np.exp(X.logW(vp - v, x0, AJ, x0, AI) + X.logW(up - u, x0, AK, x0, AI)) * star(
            [(hf - (u - v), AJ, +1), (hf - (up - vp), AK, +1), (up - v, AL, -1), (u - vp, AI, -1)])
        rat.append(F1 / F2)
    rat = np.array(rat)
    return sad, fr, np.max(abs(rat / rat[0] - 1))


if __name__ == "__main__":
    for (n, r, N) in [(3, 1, 2), (3, 2, 1), (3, 1, 3), (4, 1, 2), (3, 2, 2), (2, 1, 3)]:
        for k in range(1, n):
            for retau in [-0.2, 0.05, 0.3]:
                tau = retau + 0.35j
                sad, fr, dev = check(n, r, N, tau, k)
                print("n=%d r=%d N=%d k=%d gcd=%d Re tau=%+.2f : saddle %.1e  frac %.1e  dssr %.1e"
                      % (n, r, N, k, np.gcd(k, n), retau, sad, fr, dev), flush=True)
