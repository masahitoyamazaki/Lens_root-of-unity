"""(1) Is B^{(0)}_{u,v'}(a,b) = const / B^{(1)}_{v',u}(b,a)  (BKMS: reversed orientation = inverse)?
(2) Star-star relation with the literal BKMS weight (review arXiv:2502.17545):
    Wbar_pq(v,w) = omega^{Q(v,w)} prod_{I=1}^{|v-w|-1}(z0+(p)z0-(q) - z0+(q)z0-(p) omega^I) / prod_{i<n} prod_{I=0}^{v_i-w_i-1}(zi+(p)zi-(q) - zi+(q)zi-(p) omega^{1+I})
    with z_0^{+-} := z_n^{-+}, |v-w| = sum_{i<n}(v_i-w_i), products continued by the recursive convention;
    edges: W(l1,l2;A,B) = Wbar_{p=l2,q=l1}(A,B), except the (u,v') edges: W = 1/Wbar_{p=u,q=v'}(B,A)."""
import itertools
import numpy as np
from lensnp import PI
from bkms_pure import Qf, PureTracked
from bkms_final import logB

def prodlog(f, lo, hi):
    """log prod_{I=lo}^{hi} f(I) with the recursive convention for hi < lo-1."""
    if hi >= lo:
        return sum(f(I) for I in range(lo, hi + 1))
    return -sum(f(I) for I in range(hi + 1, lo))

def logBKMS(P, p, q, v, w):
    n, M, om = P.n, P.M, P.om
    zp, zm = P.zp, P.zm
    val = 2j * PI / M * Qf(v, w)
    d = [v[i] - w[i] for i in range(n)]
    mod = sum(d[:n - 1])
    z0p = lambda l: zm[l][n - 1]; z0m = lambda l: zp[l][n - 1]
    val += prodlog(lambda I: np.log(z0p(p) * z0m(q) - z0p(q) * z0m(p) * om ** I), 1, mod - 1)
    for i in range(n - 1):
        val -= prodlog(lambda I: np.log(zp[p][i] * zm[q][i] - zp[q][i] * zm[p][i] * om ** (1 + I)), 0, d[i] - 1)
    return val

def setup(n, M, seed, eps=0.5):
    rng = np.random.default_rng(seed)
    gdev = [eps * (rng.normal(size=4) + 1j * rng.normal(size=4)) for _ in range(n - 1)]
    base = {"u": 0.8, "up": 0.55, "v": 0.45, "vp": 0.3}
    svals = {l: np.exp(2j * base[l]) * (1 + 0.05 * (rng.normal() + 1j * rng.normal())) for l in base}
    return PureTracked(n, M, gdev, svals, steps=200), rng

def star_star(P, rng, E, trials=5):
    n, M = P.n, P.M
    rats = []
    for t in range(trials):
        S = []
        for _ in range(4):
            x = list(rng.integers(0, M, n - 1)) if t else [0] * (n - 1); x.append(-sum(x)); S.append(x)
        AI, AJ, AK, AL = S; L = R = 0
        for ah in itertools.product(range(M), repeat=n - 1):
            ah = list(ah) + [-sum(ah)]
            L += np.exp(E("v", "u", AK, ah) + E("vp", "up", AJ, ah) + E("up", "v", ah, AI) + E("u", "vp", ah, AL))
            R += np.exp(E("v", "u", ah, AJ) + E("vp", "up", ah, AK) + E("up", "v", AL, ah) + E("u", "vp", AI, ah))
        L *= np.exp(E("vp", "v", AL, AK) + E("up", "u", AL, AJ))
        R *= np.exp(E("vp", "v", AJ, AI) + E("up", "u", AK, AI))
        rats.append(L / R)
    rats = np.array(rats)
    return np.max(abs(rats / rats[0] - 1)), rats[0]

if __name__ == "__main__":
    for (n, M) in [(2, 3), (3, 3), (3, 4), (4, 2), (4, 3), (5, 2)]:
        P, rng = setup(n, M, 0)
        # (1) inversion test
        devs = []
        for _ in range(6):
            a = list(rng.integers(-3, 4, n - 1)); a.append(-sum(a)); b = list(rng.integers(-3, 4, n - 1)); b.append(-sum(b))
            devs.append(logB(P, "u", "vp", 0, a, b) + logB(P, "vp", "u", 1, b, a))
        devs = np.array(devs); dd = devs - devs[0]; dd = dd - 2j * PI * np.round(dd.imag / (2 * PI))
        # (2) literal BKMS
        def E(l1, l2, A, B):
            if (l1, l2) == ("u", "vp"):
                return -logBKMS(P, "u", "vp", B, A)
            return logBKMS(P, l2, l1, A, B)
        dev, R = star_star(P, rng, E)
        print("n=%d M=%d : (1) B0_{u,v'}(a,b) * B1_{v',u}(b,a) const? dev %.1e | (2) literal BKMS weights: star-star dev %.1e, R=%s"
              % (n, M, np.max(abs(dd)), dev, np.round(R, 6)))
