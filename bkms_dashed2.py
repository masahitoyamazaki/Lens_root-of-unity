"""Same as bkms_dashed.py but with Moebius maps near the identity (small spins) to stay on principal branches."""
import numpy as np
from lensnp import PI
from trig_general import TrigGen
import bkms_check
from bkms_dashed import saddle_res, relation
from bkms_curve4 import mob

def lines_small(n, rng, eps):
    gs = [(1, 0, 0, 1)] + [tuple(np.array([1, 0, 0, 1]) + eps * (rng.normal(size=4) + 1j * rng.normal(size=4))) for _ in range(n - 1)]
    al, be = {}, {}
    base = {"u": 0.8, "up": 0.55, "v": 0.45, "vp": 0.3}
    for l in ["u", "up", "v", "vp"]:
        s = np.exp(2j * base[l]) * (1 + 0.05 * (rng.normal() + 1j * rng.normal()))
        gam = np.log(np.array([mob(g, s) for g in gs])) / 2j
        al[l] = np.mean(gam); be[l] = gam - al[l]
    return al, be

for (n, M) in [(3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (5, 2)]:
    devs = []; ress = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        al, be = lines_small(n, rng, 0.15)
        P = [-be["up"] + be["v"], -be["up"] + be["vp"], -be["u"] + be["v"], -be["u"] + be["vp"]]
        a = [al["u"], al["up"] + PI / 2, al["v"], al["vp"] + PI / 2]
        h1 = np.zeros(n, dtype=complex); h2 = P[1] - be["u"] + be["v"]
        m = TrigGen(n, M)
        ress.append(max(saddle_res(m, P, a, h1, h2)))
        devs.append(relation(m, P, a, h1, h2, np.random.default_rng(0), trials=4))
    print("n=%d M=%d : max saddle res %.0e ; star-star devs %s" % (n, M, max(ress), ["%.0e" % d for d in devs]))
