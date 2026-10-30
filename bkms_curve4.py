"""n=3: edge rule psi_X = -beta_hor + beta_ver (psi_h=0), general offset vectors on W-type (and/or Wbar-type) neighbours."""
import itertools
import numpy as np
from lensnp import PI
from trig_general import TrigGen

def mob(m, s):
    a, b, c, d = m
    return (a * s + b) / (c * s + d)

def res(n, dW, dWb, curve, seed, sgn=1):
    rng = np.random.default_rng(seed)
    gs = [(1, 0, 0, 1)] + [tuple(rng.normal(size=4) + 1j * rng.normal(size=4)) for _ in range(n - 1)]
    al, be = {}, {}
    for l in ["u", "up", "v", "vp"]:
        s = np.exp(1j * rng.uniform(-1, 1))
        R = np.array([mob(g, s) for g in gs]) if curve else np.exp(1j * rng.uniform(-1, 1, n) + 0.1 * rng.normal(size=n))
        gam = np.log(R) / 2j
        al[l] = np.mean(gam); be[l] = gam - al[l]
    h = np.zeros(n, dtype=complex)
    Pi = h + sgn * (-be["up"] + be["v"]) + dW
    Pj = h + sgn * (-be["up"] + be["vp"]) + dWb
    Pk = h + sgn * (-be["u"] + be["v"]) + dWb
    Pl = h + sgn * (-be["u"] + be["vp"]) + dW
    u, up, v, vp = al["u"], al["up"], al["v"], al["vp"]; hf = PI / 2
    legs1 = [(hf - (u - v), Pk, -1), (hf - (up - vp), Pj, -1), (up - v, Pi, +1), (u - vp, Pl, +1)]
    g = TrigGen(n, 3).G(h, legs1); d = g[:-1] - g[-1]; d = d - 2j * PI * np.round(d.imag / (2 * PI))
    return np.max(abs(d))

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    vals = [0, PI / 2, -PI / 2, PI]
    cnt = 0
    for dW in itertools.product(vals, repeat=n):
        for dWb in [tuple([0] * n)] + list(itertools.product([0, PI / 2, -PI / 2], repeat=n)):
            for sgn in [1, -1]:
                dWv, dWbv = np.array(dW), np.array(dWb)
                rc = max(res(n, dWv, dWbv, True, s, sgn) for s in range(2))
                if rc < 1e-8:
                    rc = max(res(n, dWv, dWbv, True, s, sgn) for s in range(2, 5))
                    rn = max(res(n, dWv, dWbv, False, s, sgn) for s in range(3))
                    if rc < 1e-8 and rn > 1e-6:
                        cnt += 1
                        if cnt <= 20:
                            print("sgn=%+d dW=%s dWb=%s : curve %.1e off %.1e" % (sgn, np.round(np.array(dW) / PI, 2), np.round(np.array(dWb) / PI, 2), rc, rn))
    print("found", cnt)
