"""Step A: edge-level identification of our trigonometric weight on the BKMS curve with the BKMS form
   B(a,b) = omega^{s Q(a,b)} prod_i prod_{k=0}^{d_i-1} 1/( z_i^+(l1) z_i^-(l2) omega^{k+c} - z_i^+(l2) z_i^-(l1) ),  d = a - b,
with z_i^{+M} = a_i s + b_i, z_i^{-M} = c_i s + d_i (M-th roots chosen once per line).
Test: cross-ratio  F(a,b)F(a',b')/(F(a,b')F(a',b)) = 1  for F = ours/B  (i.e. ours/B is a gauge factor f(a)g(b))."""
import itertools
import numpy as np
from lensnp import PI
from trig_general import TrigGen

def Qf(v, w):
    n = len(v)
    return sum(v[i] * sum(v[j] - w[j] for j in range(i + 1)) for i in range(n - 1))

def setup(n, seed):
    rng = np.random.default_rng(seed)
    gs = [np.array([1, 0, 0, 1], dtype=complex)] + [np.array([1, 0, 0, 1]) + 0.5 * (rng.normal(size=4) + 1j * rng.normal(size=4)) for _ in range(n - 1)]
    ss = [np.exp(2j * x) * (1 + 0.05 * (rng.normal() + 1j * rng.normal())) for x in (0.7, 0.35)]
    return gs, ss

def our_edge(m, gs, ss, dashed, a, b):
    """our trigonometric W for the edge crossed by lines l1 (s1, dashed1) and l2 (s2, dashed2); theta = alpha1 - alpha2."""
    n = m.n
    gam = []
    for s, dsh in zip(ss, dashed):
        R = np.array([(g[0] * s + g[1]) / (g[2] * s + g[3]) for g in gs])
        gm = np.log(R) / 2j
        gam.append(gm)
    al = [np.mean(g) + (PI / 2 if d else 0) for g, d in zip(gam, dashed)]
    be = [g - np.mean(g) for g in gam]
    th = al[0] - al[1]
    PA = be[0] - be[1]; PB = np.zeros(n)
    return m.logW(th, PA, a, PB, b)

def bkms_edge(M, gs, ss, dashed, a, b, s_sign, c, order):
    n = len(gs)
    om = np.exp(2j * PI / M)
    zp, zm = [], []
    for s, dsh in zip(ss, dashed):
        num = np.array([g[0] * s + g[1] for g in gs]); den = np.array([g[2] * s + g[3] for g in gs])
        if dsh:
            num = -num          # dashed line: R -> -R
        zp.append(num ** (1.0 / M)); zm.append(den ** (1.0 / M))
    l1, l2 = (0, 1) if order == 0 else (1, 0)
    val = s_sign * 2j * PI / M * Qf(a, b)
    for i in range(n):
        d = a[i] - b[i]
        f = lambda k: np.log(zp[l1][i] * zm[l2][i] * om ** (k + c) - zp[l2][i] * zm[l1][i])
        if d >= 0:
            val -= sum(f(k) for k in range(d))
        else:
            val += sum(f(k) for k in range(d, 0))
    return val

def cross_dev(n, M, dashed, s_sign, c, order, seed=0):
    gs, ss = setup(n, seed)
    m = TrigGen(n, M)
    rng = np.random.default_rng(seed + 100)
    devs = []
    for _ in range(6):
        sp = []
        for _ in range(4):
            x = list(rng.integers(0, M, n - 1)); x.append(-sum(x)); sp.append(x)
        a, ap, b, bp = sp
        F = lambda x, y: our_edge(m, gs, ss, dashed, x, y) - bkms_edge(M, gs, ss, dashed, x, y, s_sign, c, order)
        cr = F(a, b) + F(ap, bp) - F(a, bp) - F(ap, b)
        devs.append(abs(np.exp(cr) - 1))
    return max(devs)

if __name__ == "__main__":
    for n, M in [(3, 3), (3, 4)]:
        for dashed in [(False, False), (True, False), (False, True), (True, True)]:
            hits = []
            for s_sign, c, order in itertools.product([1, -1, 0], [0, 0.5, 1], [0, 1]):
                d = cross_dev(n, M, dashed, s_sign, c, order)
                if d < 1e-9:
                    hits.append("s=%+d c=%.1f order=%d" % (s_sign, c, order))
            print("n=%d M=%d lines dashed=%s : %s" % (n, M, dashed, hits if hits else "no match"))
