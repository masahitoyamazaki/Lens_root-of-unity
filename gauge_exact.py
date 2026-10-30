"""Exact edge gauge between our (branch-tracked, verified) weights and the BKMS form (tracked roots),
in the configuration of bkms_branch.py.  For each edge e: F_e(a,b) = log ours - log BKMS = k1_e.a + k2_e.b + c_e ?"""
import numpy as np
from lensnp import PI
from continuation import Cont, edges
from bkms_branch import config
from bkms_pure import PureTracked, theta_shift

LINES = {"Elk": ("vp", "v"), "Elj": ("up", "u"), "1kh": ("v", "u"), "1jh": ("vp", "up"), "1hi": ("up", "v"), "1hl": ("u", "vp"),
         "Eji": ("vp", "v"), "Eki": ("up", "u"), "2hj": ("v", "u"), "2hk": ("vp", "up"), "2lh": ("up", "v"), "2ih": ("u", "vp")}

def build(n, M, seed, eps=0.5, steps=300):
    rng = np.random.default_rng(seed)
    gdev = [eps * (rng.normal(size=4) + 1j * rng.normal(size=4)) for _ in range(n - 1)]
    base = {"u": 0.8, "up": 0.55, "v": 0.45, "vp": 0.3}
    svals = {l: np.exp(2j * base[l]) * (1 + 0.05 * (rng.normal() + 1j * rng.normal())) for l in base}
    m = Cont(n, M); m.L = {}
    for s in range(steps + 1):
        P, a, h1, h2 = config(n, gdev, svals, s / steps)
        m.track_logs(P, h1, h2, a)
    E = {e[0]: e for e in edges(P, h1, h2, a)}
    PT = PureTracked(n, M, gdev, svals, steps=steps)
    def ours(key, A, B):
        return m.logw_key(key, E[key][1], E[key][2], E[key][3], A, B)
    def bk(key, A, B):
        l1, l2 = LINES[key]
        return PT.logB((l1,), (l2,), theta_shift(l1, l2), A, B)
    return ours, bk, m, (P, a, h1, h2), PT

if __name__ == "__main__":
    n, M = 3, 3
    ours, bk, m, cfg, PT = build(n, M, 0)
    print("sanity: tracked model satisfies star-star:", "%.1e" % m.relation(cfg[0], cfg[2], cfg[3], cfg[1], np.random.default_rng(0), trials=3))
    z = [0] * n
    basis = []
    for i in range(n - 1):
        e = [0] * n; e[i] = 1; e[n - 1] = -1; basis.append(e)
    rng = np.random.default_rng(5)
    for key in LINES:
        F = lambda A, B: ours(key, A, B) - bk(key, A, B)
        c = F(z, z)
        k1 = np.array([F(e, z) - c for e in basis]); k2 = np.array([F(z, e) - c for e in basis])
        # linearity test (mod 2 pi i)
        dev = 0
        for _ in range(5):
            A = list(rng.integers(-3, 4, n - 1)); A.append(-sum(A)); B = list(rng.integers(-3, 4, n - 1)); B.append(-sum(B))
            pred = c + np.dot(A[:n - 1], k1) + np.dot(B[:n - 1], k2)
            d = F(A, B) - pred; d = d - 2j * PI * np.round(d.imag / (2 * PI)); dev = max(dev, abs(d))
        print("%s lines=%s  k1=%s  k2=%s  k1+k2=%s  nonlinearity %.1e" % (key, LINES[key], np.round(k1, 4), np.round(k2, 4), np.round(k1 + k2, 4), dev))
