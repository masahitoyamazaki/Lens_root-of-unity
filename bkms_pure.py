"""Step B: star-star relation with pure BKMS-form weights built from points on the curve (no continuous spins):
  B_{l1 l2}(a,b) = omega^{Q(a,b)} prod_i prod_{k=0}^{a_i-b_i-1} 1/( z_i^+(l1) z_i^-(l2) omega^{k+1/2} - z_i^+(l2) z_i^-(l1) ),
  z_i^{+M} = a_i s + b_i,  z_i^{-M} = c_i s + d_i  (M-th roots fixed once per line).
Every edge of the star-star relation is W_{pi/2 + alpha_a - alpha_b}(A,B) for an ordered pair of lines (a,b):
  W_{u-v'}(h,l): (u, v'),  W_{u'-v}(h,i): (u', v),  Wbar_{u-v}(k,h): (v, u),  Wbar_{u'-v'}(j,h): (v', u'),
  ext W_{v'-v}(l,k): (v', v), W_{u'-u}(l,j): (u', u)  [with the dashed shift alpha_{u'}, alpha_{v'} += pi/2].
The extra +pi/2 relative to alpha_a - alpha_b is implemented as z^+(l1) -> e^{i pi/M} z^+(l1), and a dashed line
contributes another e^{+-i pi/M} according to its position."""
import itertools, sys
import numpy as np
from lensnp import PI

def Qf(v, w):
    n = len(v)
    return sum(v[i] * sum(v[j] - w[j] for j in range(i + 1)) for i in range(n - 1))

class Pure:
    def __init__(self, n, M, gs, svals):
        self.n, self.M = n, M
        self.om = np.exp(2j * PI / M)
        self.zp, self.zm = {}, {}
        for l, s in svals.items():
            self.zp[l] = np.array([g[0] * s + g[1] for g in gs]) ** (1.0 / M)
            self.zm[l] = np.array([g[2] * s + g[3] for g in gs]) ** (1.0 / M)

    def logB(self, l1, l2, shift1, a, b):
        """shift1 = total number of half-steps (units of pi/2 in theta) carried by line l1 relative to l2."""
        n, M, om = self.n, self.M, self.om
        zp1 = self.zp[l1[0]] * np.exp(1j * PI * shift1 / M)
        zp2 = self.zp[l2[0]]
        zm1, zm2 = self.zm[l1[0]], self.zm[l2[0]]
        val = 2j * PI / M * Qf(a, b)
        for i in range(n):
            d = a[i] - b[i]
            f = lambda k: np.log(zp1[i] * zm2[i] * om ** (k + 0.5) - zp2[i] * zm1[i])
            val += -sum(f(k) for k in range(d)) if d >= 0 else sum(f(k) for k in range(d, 0))
        return val

def theta_shift(l1, l2, wbar=None):
    """number of pi/2 units in theta relative to (raw alpha_l1 - alpha_l2):
    W edges: dashed(l1) - dashed(l2);  Wbar edges (theta = pi/2 - ...): 1 + dashed(l1) - dashed(l2).
    Wbar edges of the star-star relation are exactly those with lines (v,u) or (vp,up)."""
    dsh = {"u": 0, "v": 0, "up": 1, "vp": 1}
    if wbar is None:
        wbar = (l1, l2) in [("v", "u"), ("vp", "up")]
    return (1 if wbar else 0) + dsh[l1] - dsh[l2]

def run(n, M, seed, trials=5, eps=0.5):
    rng = np.random.default_rng(seed)
    gs = [np.array([1, 0, 0, 1], dtype=complex)] + [np.array([1, 0, 0, 1]) + eps * (rng.normal(size=4) + 1j * rng.normal(size=4)) for _ in range(n - 1)]
    base = {"u": 0.8, "up": 0.55, "v": 0.45, "vp": 0.3}
    svals = {l: np.exp(2j * base[l]) * (1 + 0.05 * (rng.normal() + 1j * rng.normal())) for l in base}
    P = Pure(n, M, gs, svals)
    E = lambda l1, l2, A, B: P.logB((l1,), (l2,), theta_shift(l1, l2), A, B)
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
    for (n, M) in [(2, 3), (2, 5), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (5, 2)]:
        res = [run(n, M, s) for s in range(3)]
        print("n=%d M=%d : pure-BKMS-form star-star devs %s" % (n, M, ["%.0e" % r[0] for r in res]))


class PureTracked(Pure):
    """z_i^{+-}(l) = M-th roots continued from g_i = identity (common root for all components)."""
    def __init__(self, n, M, gdev, svals, steps=400):
        self.n, self.M = n, M
        self.om = np.exp(2j * PI / M)
        self.zp, self.zm = {}, {}
        for l, s in svals.items():
            lp = np.full(n, np.log(s) / M, dtype=complex); lm = np.zeros(n, dtype=complex)
            for st in range(1, steps + 1):
                lam = st / steps
                gs = [np.array([1, 0, 0, 1], dtype=complex)] + [np.array([1, 0, 0, 1]) + lam * d for d in gdev]
                num = np.array([g[0] * s + g[1] for g in gs]); den = np.array([g[2] * s + g[3] for g in gs])
                for arr, val in ((lp, num), (lm, den)):
                    new = np.log(val) / M
                    k = np.round(((arr - new) * M).imag / (2 * PI))
                    arr[:] = new + 2j * PI * k / M
            self.zp[l] = np.exp(lp); self.zm[l] = np.exp(lm)

def run_tracked(n, M, seed, trials=5, eps=0.5):
    rng = np.random.default_rng(seed)
    gdev = [eps * (rng.normal(size=4) + 1j * rng.normal(size=4)) for _ in range(n - 1)]
    base = {"u": 0.8, "up": 0.55, "v": 0.45, "vp": 0.3}
    svals = {l: np.exp(2j * base[l]) * (1 + 0.05 * (rng.normal() + 1j * rng.normal())) for l in base}
    P = PureTracked(n, M, gdev, svals)
    E = lambda l1, l2, A, B: P.logB((l1,), (l2,), theta_shift(l1, l2), A, B)
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
