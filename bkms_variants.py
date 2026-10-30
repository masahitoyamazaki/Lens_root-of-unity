"""Scan conventions for the numerator (0-th component) of the BKMS weight; denominators and omega^Q as in the review."""
import itertools
import numpy as np
from lensnp import PI
from bkms_pure import Qf
from bkms_literal import prodlog, setup, star_star

def make_logW(P, swap, wpos, sgn, lo):
    n, M, om = P.n, P.M, P.om
    zp, zm = P.zp, P.zm
    def logW(p, q, v, w):
        val = 2j * PI / M * Qf(v, w)
        d = [v[i] - w[i] for i in range(n)]
        mod = sum(d[:n - 1])
        z0p = (lambda l: zm[l][n - 1]) if swap else (lambda l: zp[l][n - 1])
        z0m = (lambda l: zp[l][n - 1]) if swap else (lambda l: zm[l][n - 1])
        if wpos == 2:
            f = lambda I: np.log(z0p(p) * z0m(q) - z0p(q) * z0m(p) * om ** (sgn * I))
        else:
            f = lambda I: np.log(z0p(p) * z0m(q) * om ** (sgn * I) - z0p(q) * z0m(p))
        val += prodlog(f, lo, mod - 1 + lo - 1 + 1 - 1) if False else prodlog(f, lo, lo + mod - 1 - (1 if lo == 1 else 0))
        for i in range(n - 1):
            val -= prodlog(lambda I: np.log(zp[p][i] * zm[q][i] - zp[q][i] * zm[p][i] * om ** (1 + I)), 0, d[i] - 1)
        return val
    return logW

if __name__ == "__main__":
    for (n, M) in [(3, 3), (3, 4), (2, 5)]:
        print("n=%d M=%d" % (n, M))
        for swap, wpos, sgn, lo in itertools.product([True, False], [1, 2], [1, -1], [0, 1]):
            devs = []
            for seed in range(2):
                P, rng = setup(n, M, seed)
                W = make_logW(P, swap, wpos, sgn, lo)
                def E(l1, l2, A, B):
                    if (l1, l2) == ("u", "vp"):
                        return -W("u", "vp", B, A)
                    return W(l2, l1, A, B)
                devs.append(star_star(P, rng, E, trials=4)[0])
            if max(devs) < 1e-9:
                print("   OK: z0 swapped=%s, omega on term %d, omega^(%+d I), I from %d : dev %.1e" % (swap, wpos, sgn, lo, max(devs)))
