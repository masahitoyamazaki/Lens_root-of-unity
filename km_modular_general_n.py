"""Proof of the modular covariance (c) of (KMnssr) for general n (draft, 'Origin of the general torsion points').
All transformation exponents are polynomials in d=a_i-b_j and q=i-j (edges) or in h_i-h_j and j-i (vertex). Sums over components are
expressed through the moments  mu_X(s,t) = sum_i X_i^s i^t  of each spin X, with sum_i i^t a polynomial in the symbol n and mu_X(1,0)=0.
Treating the remaining moments as independent symbols (stronger than needed), we show that the total exponent of each side of (KMnssr)
does not depend on the moments of the internal spin, and that the difference of the two sides does not depend on the external spins."""
import sympy as sp

n, M, w0, ga, J, Tp = sp.symbols('n M omega0 gamma J Tp')
u, up, v, vp = sp.symbols("u u' v v'")
pi, I = sp.pi, sp.I
k, d, q = sp.symbols('k d q')
half = sp.Rational(1, 2)


def faulhaber(t):
    i = sp.Symbol('i', integer=True)
    return sp.expand(sp.summation(i**t, (i, 0, n - 1)))


def factor_exp(XM, Y, mp):
    return -I * ga * XM**2 / (pi * J) + 2 * I * mp * Y - I * pi * mp**2 * Tp


def edge_poly(theta_e, barred):
    """sum_{k<d} of minus the factor exponent, as a polynomial in d, q."""
    D = q * w0
    XM = (D + theta_e + (k + half) * pi) / M
    th_p = (pi / 2 - (pi / 2 - theta_e) / J) if barred else theta_e / J
    Y = (D / J + th_p + (k + half) * pi) / M
    mp = (k + (1 if barred else half)) * ga / M
    f = -factor_exp(XM, Y, mp)
    return sp.expand(sp.summation(sp.expand(f), (k, 0, d - 1)))


def vertex_poly():
    """exponent of one vertex factor, as a polynomial in x=h_i-h_j and p=j-i (symmetric under (x,p)->(-x,-p))."""
    x, p = sp.symbols('x p')
    Z = (pi * x - p * w0) / M
    Y = (pi * x - p * w0 / J) / M
    return sp.expand(factor_exp(Z, Y, x * ga / M)), x, p


class Moments:
    def __init__(self, name):
        self.name = name; self.cache = {}

    def __call__(self, s, t):
        if s == 0:
            return faulhaber(t)
        if (s, t) == (1, 0):
            return 0
        key = (s, t)
        if key not in self.cache:
            self.cache[key] = sp.Symbol('%s_%d_%d' % (self.name, s, t))
        return self.cache[key]


def double_sum(poly, var1, var2, A, B, sign_q=1):
    """sum_{i,j} poly(var1 = A_i - B_j, var2 = sign_q*(i - j))."""
    P = sp.Poly(poly, var1, var2)
    tot = 0
    for (pp, rr), c in P.terms():
        for s_ in range(pp + 1):
            for t_ in range(rr + 1):
                coef = sp.binomial(pp, s_) * sp.binomial(rr, t_) * (-1)**(pp - s_) * (-1)**(rr - t_) * sign_q**rr
                tot += c * coef * A(s_, t_) * B(pp - s_, rr - t_)
    return sp.expand(tot)


def edge(theta_e, barred, A, B, cache={}):
    key = (theta_e, barred)
    if key not in cache:
        cache[key] = edge_poly(theta_e, barred)
    return double_sum(cache[key], d, q, A, B)


def vertex(H):
    G, x, p = vertex_poly()
    # sum_{i<j} G(h_i-h_j, j-i) = 1/2 sum_{i,j} G(h_i-h_j, -(i-j))  (G symmetric, diagonal terms vanish)
    return sp.expand(double_sum(G, x, p, H, H, sign_q=-1) / 2)


if __name__ == "__main__":
    S = {X: Moments(X) for X in "IJKLH"}
    hf = pi / 2
    W = lambda t, A, B: edge(t, False, A, B)
    Wb = lambda t, A, B: edge(hf - t, True, A, B)
    I_, J_, K_, L_, H_ = (S[x] for x in "IJKLH")
    lhs = sp.expand(W(vp - v, L_, K_) + W(up - u, L_, J_) + vertex(H_) + Wb(u - v, K_, H_) + Wb(up - vp, J_, H_) + W(up - v, H_, I_) + W(u - vp, H_, L_))
    rhs = sp.expand(W(vp - v, J_, I_) + W(up - u, K_, I_) + vertex(H_) + Wb(u - v, H_, J_) + Wb(up - vp, H_, K_) + W(up - v, L_, H_) + W(u - vp, I_, H_))
    hsyms = list(H_.cache.values())
    print("internal-spin moments:", hsyms)
    print("LHS depends on them:", [sp.simplify(sp.diff(lhs, h)) for h in hsyms])
    print("RHS depends on them:", [sp.simplify(sp.diff(rhs, h)) for h in hsyms])
    ext = [s_ for X in "IJKL" for s_ in S[X].cache.values()]
    diff = sp.expand(lhs - rhs)
    print("LHS-RHS depends on external moments:", [sp.simplify(sp.diff(diff, e)) for e in ext])
