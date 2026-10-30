"""Literal review formula (omega^{+I} numerator, omega^{1+I} denominator, omega^{sQ}) under all identifications:
 (p,q) = (l2,l1) or (l1,l2);  (v,w) = (A,B) or (B,A);  sQ = +-1;  conjugation omega -> omega^{-1}; special edge as inverse."""
import itertools
import numpy as np
from lensnp import PI
from bkms_pure import Qf
from bkms_literal import prodlog, setup, star_star

def make(P, sQ, conj, numsign):
    n, M = P.n, P.M
    om = np.exp(-2j * PI / M) if conj else np.exp(2j * PI / M)
    zp, zm = P.zp, P.zm
    def W(p, q, v, w):
        val = sQ * (np.log(om)) * Qf(v, w)
        d = [v[i] - w[i] for i in range(n)]; mod = sum(d[:n - 1])
        z0p = lambda l: zm[l][n - 1]; z0m = lambda l: zp[l][n - 1]
        val += prodlog(lambda I: np.log(z0p(p) * z0m(q) - z0p(q) * z0m(p) * om ** (numsign * I)), 1, mod - 1)
        for i in range(n - 1):
            val -= prodlog(lambda I: np.log(zp[p][i] * zm[q][i] - zp[q][i] * zm[p][i] * om ** (1 + I)), 0, d[i] - 1)
        return val
    return W

for (n, M) in [(3, 3), (3, 4)]:
    print("n=%d M=%d" % (n, M))
    for pq, vw, sQ, conj, numsign in itertools.product(["l2l1", "l1l2"], ["AB", "BA"], [1, -1], [False, True], [1, -1]):
        devs = []
        for seed in range(2):
            P, rng = setup(n, M, seed)
            W = make(P, sQ, conj, numsign)
            def E(l1, l2, A, B, W=W):
                inv = (l1, l2) == ("u", "vp")
                if inv:
                    l1, l2, A, B = l2, l1, B, A
                p, q = (l2, l1) if pq == "l2l1" else (l1, l2)
                v, w = (A, B) if vw == "AB" else (B, A)
                r = W(p, q, v, w)
                return -r if inv else r
            devs.append(star_star(P, rng, E, trials=4)[0])
        if max(devs) < 1e-9:
            print("   OK: (p,q)=%s (v,w)=%s sQ=%+d conj=%s numerator omega^(%+d I)" % (pq, vw, sQ, conj, numsign))
