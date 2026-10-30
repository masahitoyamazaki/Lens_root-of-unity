"""Branch-consistent check: deform the Moebius maps from the identity (lam=0, FZ point) to the target (lam=1),
with the Z-invariant saddles constructed exactly at every step; track log(2cos x) of every (edge, component)."""
import numpy as np
from lensnp import PI
from continuation import Cont
from bkms_curve4 import mob

def config(n, gdev, svals, lam):
    gs = [(1, 0, 0, 1)] + [tuple(np.array([1, 0, 0, 1]) + lam * d) for d in gdev]
    al, be = {}, {}
    for l in ["u", "up", "v", "vp"]:
        gam = np.log(np.array([mob(g, svals[l]) for g in gs])) / 2j
        al[l] = np.mean(gam); be[l] = gam - al[l]
    P = [-be["up"] + be["v"], -be["up"] + be["vp"], -be["u"] + be["v"], -be["u"] + be["vp"]]
    a = [al["u"], al["up"] + PI / 2, al["v"], al["vp"] + PI / 2]
    return P, a, np.zeros(n, dtype=complex), P[1] - be["u"] + be["v"]

def run(n, M, seed, eps=0.15, steps=300):
    rng = np.random.default_rng(seed)
    gdev = [eps * (rng.normal(size=4) + 1j * rng.normal(size=4)) for _ in range(n - 1)]
    base = {"u": 0.8, "up": 0.55, "v": 0.45, "vp": 0.3}
    svals = {l: np.exp(2j * base[l]) * (1 + 0.05 * (rng.normal() + 1j * rng.normal())) for l in base}
    m = Cont(n, M)
    m.L = {}
    for s in range(steps + 1):
        P, a, h1, h2 = config(n, gdev, svals, s / steps)
        m.track_logs(P, h1, h2, a)
    return m.relation(P, h1, h2, a, np.random.default_rng(0), trials=4)

if __name__ == "__main__":
    for (n, M) in [(3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (5, 2)]:
        print("n=%d M=%d : branch-tracked star-star devs %s" % (n, M, ["%.0e" % run(n, M, s) for s in range(4)]))
