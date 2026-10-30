"""(a) non-Moebius control; (b) full discrete star-star relation at BKMS-curve-parametrised spins."""
import itertools
import numpy as np
from lensnp import PI
from trig_general import TrigGen
from bkms_curve4 import mob

def lines(n, rng, kind):
    gs = [(1, 0, 0, 1)] + [tuple(rng.normal(size=4) + 1j * rng.normal(size=4)) for _ in range(n - 1)]
    al, be = {}, {}
    for l in ["u", "up", "v", "vp"]:
        s = np.exp(1j * rng.uniform(-1, 1))
        if kind == "mobius":
            R = np.array([mob(g, s) for g in gs])
        else:   # non-Moebius: R_i = s^(i) + c_i
            R = np.array([s ** (i + 1) + 0.3 * i for i in range(n)])
        gam = np.log(R) / 2j
        al[l] = np.mean(gam); be[l] = gam - al[l]
    return al, be

def config(n, al, be):
    dW = np.full(n, PI / 2)
    Pi = -be["up"] + be["v"] + dW; Pj = -be["up"] + be["vp"]
    Pk = -be["u"] + be["v"]; Pl = -be["u"] + be["vp"] + dW
    return Pi, Pj, Pk, Pl

if __name__ == "__main__":
    # (a)
    for n in [2, 3]:
        r = []
        for s in range(4):
            rng = np.random.default_rng(s)
            al, be = lines(n, rng, "power")
            Pi, Pj, Pk, Pl = config(n, al, be)
            u, up, v, vp = al["u"], al["up"], al["v"], al["vp"]; hf = PI / 2
            legs1 = [(hf - (u - v), Pk, -1), (hf - (up - vp), Pj, -1), (up - v, Pi, +1), (u - vp, Pl, +1)]
            g = TrigGen(n, 3).G(np.zeros(n), legs1); d = g[:-1] - g[-1]; d = d - 2j * PI * np.round(d.imag / (2 * PI))
            r.append(np.max(abs(d)))
        print("(a) n=%d non-Moebius relation: saddle residual min %.1e" % (n, min(r)))
    # (b) full discrete star-star relation; external spins from the curve, saddles solved numerically
    for (n, M) in [(3, 2), (3, 3), (4, 2)]:
        rng = np.random.default_rng(7)
        al, be = lines(n, rng, "mobius")
        Pi, Pj, Pk, Pl = config(n, al, be)
        u, up, v, vp = al["u"], al["up"], al["v"], al["vp"]; hf = PI / 2
        m = TrigGen(n, M, cplx=True)
        legs1 = [(hf - (u - v), Pk, -1), (hf - (up - vp), Pj, -1), (up - v, Pi, +1), (u - vp, Pl, +1)]
        legs2 = [(hf - (u - v), Pj, +1), (hf - (up - vp), Pk, +1), (up - v, Pl, -1), (u - vp, Pi, -1)]
        h1, e1 = m.saddle_c(legs1, np.full(n, 0.01))
        h2, e2 = m.saddle_c(legs2, h1)
        rats = []
        for t in range(4):
            A = []
            for _ in range(4):
                a = list(rng.integers(0, M, n - 1)) if t else [0] * (n - 1); a.append(-sum(a)); A.append(a)
            AI, AJ, AK, AL = A; L = R = 0
            for ah in itertools.product(range(M), repeat=n - 1):
                ah = list(ah) + [-sum(ah)]
                L += np.exp(m.logS(h1, ah) + m.logW(hf - (u - v), Pk, AK, h1, ah) + m.logW(hf - (up - vp), Pj, AJ, h1, ah)
                            + m.logW(up - v, h1, ah, Pi, AI) + m.logW(u - vp, h1, ah, Pl, AL))
                R += np.exp(m.logS(h2, ah) + m.logW(hf - (u - v), h2, ah, Pj, AJ) + m.logW(hf - (up - vp), h2, ah, Pk, AK)
                            + m.logW(up - v, Pl, AL, h2, ah) + m.logW(u - vp, Pi, AI, h2, ah))
            L *= np.exp(m.logW(vp - v, Pl, AL, Pk, AK) + m.logW(up - u, Pl, AL, Pj, AJ))
            R *= np.exp(m.logW(vp - v, Pj, AJ, Pi, AI) + m.logW(up - u, Pk, AK, Pi, AI))
            rats.append(L / R)
        rats = np.array(rats)
        print("(b) n=%d M=%d: |h1 - 0| = %.1e (res %.0e), h2 res %.0e, star-star dev %.1e"
              % (n, M, np.max(abs(h1 - np.round(h1.real / PI) * PI)), e1, e2, np.max(abs(rats / rats[0] - 1))))
