"""Symbolic form of the trigonometric saddle point equations of star 1 with Z-invariant spins (dictionary of Thm 5.8).
E_p := exp(G_p) = prod over legs, components i of:
   i == p : (2 cos x)^{+-1};   i != p : exp(-+ i x) according to the sign of c   (constants dropped).
We return, for each p, the list of cos-arguments (with exponents) and the linear exponent, in terms of gamma_{l,i}."""
import sympy as sp

def build(n):
    lines = ["u", "up", "v", "vp"]
    g = {l: [sp.Symbol("g_%s%d" % (l, i + 1)) for i in range(n)] for l in lines}
    mean = {l: sum(g[l]) / n for l in lines}
    al = {l: mean[l] + (sp.pi / 2 if l in ("up", "vp") else 0) for l in lines}
    be = {l: [g[l][i] - mean[l] for i in range(n)] for l in lines}
    vec = lambda f: [sp.expand(f(i)) for i in range(n)]
    Pi = vec(lambda i: -be["up"][i] + be["v"][i]); Pj = vec(lambda i: -be["up"][i] + be["vp"][i])
    Pk = vec(lambda i: -be["u"][i] + be["v"][i]);  Pl = vec(lambda i: -be["u"][i] + be["vp"][i])
    hf = sp.pi / 2
    u, up, v, vp = al["u"], al["up"], al["v"], al["vp"]
    legs = [(hf - (u - v), Pk, -1), (hf - (up - vp), Pj, -1), (up - v, Pi, +1), (u - vp, Pl, +1)]
    out = []
    for p in range(n):
        cosargs = []; lin = 0
        for th, X, sg in legs:
            for i in range(n):
                if sg > 0:
                    x = sp.expand(0 - X[i] + th); c = sp.Rational(p - i, n)
                else:
                    x = sp.expand(X[i] - 0 + th); c = sp.Rational(i - p, n)
                if c == 0:
                    cosargs.append((x, sg))
                else:
                    term = -sp.I * x if c > 0 else sp.I * x
                    lin += sg * term
        out.append((cosargs, sp.expand(lin)))
    return g, out

if __name__ == "__main__":
    for n in [2, 3]:
        g, out = build(n)
        print("==== n =", n)
        for p, (ca, lin) in enumerate(out):
            print(" p=%d  cos-args:" % (p + 1), [(sp.simplify(x), e) for x, e in ca])
            print("       linear exponent:", sp.simplify(lin))
        for p in range(n - 1):
            print(" lin_%d - lin_%d =" % (p + 1, p + 2), sp.simplify(out[p][1] - out[p + 1][1]))
