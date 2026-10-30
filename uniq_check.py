"""(a) star 2 symbolic structure; (b) numerical check: trig saddle residual of star 1 at Z-invariant spins equals
 the p-dependence of the cross-ratio CR_p = -(R_up-R_v)(R_u-R_vp)/((R_v-R_u)(R_vp-R_up)) (component p)."""
import numpy as np, sympy as sp
from lensnp import PI
from trig_general import TrigGen

def star2(n):
    lines = ["u", "up", "v", "vp"]
    g = {l: [sp.Symbol("g_%s%d" % (l, i + 1)) for i in range(n)] for l in lines}
    mean = {l: sum(g[l]) / n for l in lines}
    al = {l: mean[l] + (sp.pi / 2 if l in ("up", "vp") else 0) for l in lines}
    be = {l: [g[l][i] - mean[l] for i in range(n)] for l in lines}
    vec = lambda f: [sp.expand(f(i)) for i in range(n)]
    Pi = vec(lambda i: -be["up"][i] + be["v"][i]); Pj = vec(lambda i: -be["up"][i] + be["vp"][i])
    Pk = vec(lambda i: -be["u"][i] + be["v"][i]);  Pl = vec(lambda i: -be["u"][i] + be["vp"][i])
    h = vec(lambda i: Pj[i] - be["u"][i] + be["v"][i])
    hf = sp.pi / 2; u, up, v, vp = al["u"], al["up"], al["v"], al["vp"]
    legs = [(hf - (u - v), Pj, +1), (hf - (up - vp), Pk, +1), (up - v, Pl, -1), (u - vp, Pi, -1)]
    out = []
    for p in range(n):
        ca = []; lin = 0
        for th, X, sg in legs:
            for i in range(n):
                if sg > 0:
                    x = sp.expand(h[p] - X[i] + th); c = sp.Rational(p - i, n)
                else:
                    x = sp.expand(X[i] - h[p] + th); c = sp.Rational(i - p, n)
                if c == 0: ca.append((sp.simplify(x), sg))
                else: lin += sg * (-sp.I * x if c > 0 else sp.I * x)
        out.append((ca, sp.simplify(lin)))
    return out

if __name__ == "__main__":
    out = star2(3)
    for p, (ca, lin) in enumerate(out):
        print("star2 p=%d cos-args:" % (p + 1), ca, " lin:", lin)
    # (b) numerics with random (non-curve) line data
    for n in [3, 4]:
        rng = np.random.default_rng(1)
        for trial in range(3):
            R = {l: np.exp(1j * rng.uniform(-1, 1, n) + 0.2 * rng.normal(size=n)) for l in ["u", "up", "v", "vp"]}
            gam = {l: np.log(R[l]) / 2j for l in R}
            al = {l: np.mean(gam[l]) for l in R}; be = {l: gam[l] - al[l] for l in R}
            P = [-be["up"] + be["v"], -be["up"] + be["vp"], -be["u"] + be["v"], -be["u"] + be["vp"]]
            a = [al["u"], al["up"] + PI / 2, al["v"], al["vp"] + PI / 2]; hf = PI / 2
            u, up, v, vp = a
            legs1 = [(hf - (u - v), P[2], -1), (hf - (up - vp), P[1], -1), (up - v, P[0], +1), (u - vp, P[3], +1)]
            G = TrigGen(n, 3).G(np.zeros(n), legs1)
            eG = np.exp(G - G[-1])
            CR = -(R["up"] - R["v"]) * (R["u"] - R["vp"]) / ((R["v"] - R["u"]) * (R["vp"] - R["up"]))
            print("n=%d trial %d: exp(G_p - G_n) = %s ;  CR_p/CR_n = %s" % (n, trial, np.round(eG, 6), np.round(CR / CR[-1], 6)))
