"""f_e(a) = F_e(a,0) - F_e(0,0): fit Im f = lin + (pi/(2M)) * (c11 a1^2 + c22 a2^2 + c12 a1 a2)  (n=3, a3=-a1-a2), mod 2 pi.
Also report whether Re f is linear."""
import itertools
import numpy as np
from lensnp import PI
from gauge_exact import build, LINES

n, M = 3, 3
ours, bk, m, cfg, PT = build(n, M, 0)
z = [0] * n
pts = [(x, y) for x in range(-3, 4) for y in range(-3, 4)]
def wrap(x): return x - 2 * PI * np.round(x / (2 * PI))
for key in ["1kh", "1hi", "Elk"]:
    for side in ["first", "second"]:
        F = lambda A: (ours(key, A, z) - bk(key, A, z)) if side == "first" else (ours(key, z, A) - bk(key, z, A))
        c = F(z)
        vals = {p: F([p[0], p[1], -p[0] - p[1]]) - c for p in pts}
        # real part linearity
        r1, r2 = vals[(1, 0)].real, vals[(0, 1)].real
        relin = max(abs(vals[p].real - r1 * p[0] - r2 * p[1]) for p in pts)
        best = None
        for c11, c22, c12 in itertools.product(range(-4 * M, 4 * M + 1), repeat=3):
            q = lambda x, y: PI / (2 * M) * (c11 * x * x + c22 * y * y + c12 * x * y)
            l1 = wrap(vals[(1, 0)].imag - q(1, 0)); l2 = wrap(vals[(0, 1)].imag - q(0, 1))
            res = max(abs(wrap(vals[p].imag - q(*p) - l1 * p[0] - l2 * p[1])) for p in pts)
            if best is None or res < best[0]:
                best = (res, (c11, c22, c12), (l1, l2))
            if res < 1e-8:
                break
        print("%s (%s spin): Re linear dev %.1e ; Im quad coeffs (units pi/2M) %s, lin %s, resid %.1e"
              % (key, side, relin, best[1], np.round(best[2], 4), best[0]))
