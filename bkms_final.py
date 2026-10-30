"""Final form: BKMS weights with integer offsets, raw curve coordinates z_i^{+M}=a_i s+b_i, z_i^{-M}=c_i s+d_i:
  B^{(c)}_{l1 l2}(a,b) = omega^{Q(a,b)} prod_i prod_{I=0}^{a_i-b_i-1} 1/( z_i^+(l1) z_i^-(l2) omega^{I+c} - z_i^+(l2) z_i^-(l1) ),
  c = 1 for all edges except W_{u-v'} (lines (u,v')), where c = 0.  Vertex weight 1, R = 1."""
import itertools
import numpy as np
from lensnp import PI
from bkms_pure import Qf, PureTracked

def logB(P, l1, l2, c, a, b):
    n, M, om = P.n, P.M, P.om
    val = 2j * PI / M * Qf(a, b)
    for i in range(n):
        d = a[i] - b[i]
        f = lambda k: np.log(P.zp[l1][i] * P.zm[l2][i] * om ** (k + c) - P.zp[l2][i] * P.zm[l1][i])
        val += -sum(f(k) for k in range(d)) if d >= 0 else sum(f(k) for k in range(d, 0))
    return val

def run(n, M, seed, principal=False, trials=5, eps=0.5):
    rng = np.random.default_rng(seed)
    gdev = [eps * (rng.normal(size=4) + 1j * rng.normal(size=4)) for _ in range(n - 1)]
    base = {"u": 0.8, "up": 0.55, "v": 0.45, "vp": 0.3}
    svals = {l: np.exp(2j * base[l]) * (1 + 0.05 * (rng.normal() + 1j * rng.normal())) for l in base}
    P = PureTracked(n, M, gdev, svals, steps=200)
    if principal:
        gs = [np.array([1, 0, 0, 1], dtype=complex)] + [np.array([1, 0, 0, 1]) + d for d in gdev]
        for l, s in svals.items():
            P.zp[l] = np.array([g[0] * s + g[1] for g in gs]) ** (1.0 / M)
            P.zm[l] = np.array([g[2] * s + g[3] for g in gs]) ** (1.0 / M)
    E = lambda l1, l2, A, B: logB(P, l1, l2, 0 if (l1, l2) == ("u", "vp") else 1, A, B)
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
        t = [run(n, M, s) for s in range(3)]
        p = [run(n, M, s, principal=True) for s in range(3)]
        print("n=%d M=%d : tracked roots devs %s R=%s | principal roots devs %s"
              % (n, M, ["%.0e" % r[0] for r in t], ["%.3f" % abs(r[1]) for r in t], ["%.0e" % r[0] for r in p]))
