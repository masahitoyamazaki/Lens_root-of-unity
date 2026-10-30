"""Modular covariance of the elliptic Kashiwara-Miwa type star-star relation (KMnW, KMnS, KMnssr) (draft Sec. 5.3).
gamma=[[a,b],[c,d]] with a=1 mod 2M, c=0 mod 2M:  (T, omega0, theta) -> (T'=gamma T, omega0/J, theta/J),  J=cT+d.
For each star of (KMnssr) we compute the total log-ratio  [T-model] - [T'-model]  (vertex + four edges, theta'=theta/J,
crossing pi/2 in both frames) and check (a) it is independent of the internal spin h, (b) LHS-RHS totals (including the two
external edges) do not depend on the external spins.  Then (KMnssr) at (T,omega0) <=> (KMnssr) at (T',omega0/J)."""
import itertools, numpy as np, warnings
warnings.filterwarnings("ignore")
from lensnp import PI
from quano_fujii import wrap
from qf_check import logW, logS, spins_of


def covariance(n, M, T, w0, g, seed=0):
    a, b, c, d = g
    J = c * T + d; Tp = (a * T + b) / J; w0p = w0 / J
    E = lambda th, A, B: logW(n, M, T, w0, th, A, B) - logW(n, M, Tp, w0p, th / J, A, B)
    Eb = lambda th, A, B: logW(n, M, T, w0, PI / 2 - th, A, B) - logW(n, M, Tp, w0p, PI / 2 - th / J, A, B)
    Sg = lambda A: logS(n, M, T, w0, A) - logS(n, M, Tp, w0p, A)
    spins = spins_of(n, M); rng = np.random.default_rng(seed)
    u, up, v, vp = rng.uniform(0, 1, 4)
    hdev = 0; ext = []
    for t in range(4):
        I, Jx, K, L = [spins[x] for x in rng.integers(0, len(spins), 4)]
        t1 = [Sg(h) + Eb(u - v, K, h) + Eb(up - vp, Jx, h) + E(up - v, h, I) + E(u - vp, h, L) for h in spins]
        t2 = [Sg(h) + Eb(u - v, h, Jx) + Eb(up - vp, h, K) + E(up - v, L, h) + E(u - vp, I, h) for h in spins]
        hdev = max(hdev, max(abs(wrap(x - t1[0])) for x in t1), max(abs(wrap(x - t2[0])) for x in t2))
        ext.append(E(vp - v, L, K) + E(up - u, L, Jx) + t1[0] - E(vp - v, Jx, I) - E(up - u, K, I) - t2[0])
    edev = max(abs(wrap(x - ext[0])) for x in ext)
    return hdev, edev


if __name__ == "__main__":
    T = 0.13 + 0.55j
    for n, M, (kap, nu), g in [(3, 2, (1, 1), (1, 0, 4, 1)), (3, 2, (1, 0), (1, 0, 4, 1)), (3, 4, (1, 1), (1, 0, 8, 1)),
                                (5, 2, (1, 2), (1, 0, 4, 1)), (3, 3, (1, 0), (1, 0, 6, 1)), (3, 2, (1, 1), (5, 1, 4, 1)),
                                (2, 3, (1, 1), (1, 0, 6, 1)), (4, 3, (1, 1), (1, 0, 6, 1))]:
        w0 = PI * (kap * M * T - nu) / n
        hdev, edev = covariance(n, M, T, w0, g)
        X, Y = (-nu) % n, kap % n
        a, b, c, d = g
        Xp, Yp = (a * X - b * M * Y) % n, (-(c // M) * X + d * Y) % n
        print("n=%d M=%d (kappa,nu)=(%d,%d) -> (kappa',nu')=(%d,%d) gamma=%s : h-dependence %.1e, external dependence %.1e"
              % (n, M, kap, nu, Yp, (-Xp) % n, g, hdev, edev))
