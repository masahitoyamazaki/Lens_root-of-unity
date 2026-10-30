"""Symbolic proof of the modular covariance (c) of (KMnssr) (draft Sec. 'Origin of the general torsion points').
g=(alpha beta; gamma delta), alpha=1 mod 2M, gamma=0 mod 2M, J=gamma T+delta, T'=gT, omega0'=omega0/J, theta'=theta/J.
For every theta-factor  thbar1(X/M|T) = (const) exp(-i gamma (X/M)^2/(pi J)) * exp(2i m' Y - i pi m'^2 T') * (-1)^(m+m') * thbar1(Y|T'),
where Y is the corresponding argument of the T'-model and X/(MJ) = Y + m pi - m' pi T'.  The signs and constants cancel (see text);
here we show that the remaining exponents, summed over (KMnssr), are independent of the internal spin and of the external spins."""
import itertools
import sympy as sp

T, w0, al, ga, de, th = sp.symbols('T omega0 alpha gamma delta theta')
u, up, v, vp = sp.symbols("u u' v v'")
pi, I = sp.pi, sp.I
J = ga * T + de
Tp = sp.Symbol("Tp")          # T' ; only enters through m'^2 T' and is kept as an independent symbol


def factor_exp(X_over_M, Y, m_prime):
    """exponent of thbar1(X/M|T)/thbar1(Y|T') without signs and constants"""
    return -I * ga * X_over_M**2 / (pi * J) + 2 * I * m_prime * Y - I * pi * m_prime**2 * Tp


def edge_exp(n, M, A, B, theta_e, c_e, kk):
    """log W_T(theta_e;A,B) - log W_{T'}(theta_e';A,B) summed symbolically over k < d = A_i - B_j (polynomial continuation)."""
    tot = 0
    for i in range(n):
        for j in range(n):
            d = A[i] - B[j]
            D = (i - j) * w0
            X = (D + theta_e + (kk + sp.Rational(1, 2)) * pi) / M
            # T'-model argument: (i-j) w0/J + theta_e' + (k+1/2) pi, with theta_e' = theta_e/J for W, pi/2 - theta/J for Wbar
            Y = (D / J + (theta_e / J if c_e == sp.Rational(1, 2) else pi / 2 - (pi / 2 - theta_e) / J) + (kk + sp.Rational(1, 2)) * pi) / M
            mp = (kk + c_e) * ga / M
            f = -factor_exp(X, Y, mp)          # W = prod 1/thbar1
            tot += sp.summation(sp.expand(f), (kk, 0, d - 1))
    return tot


def vertex_exp(n, M, H):
    tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            hij = H[i] - H[j]
            Z = (pi * hij - (j - i) * w0) / M
            Y = (pi * hij - (j - i) * w0 / J) / M
            mp = hij * ga / M
            tot += factor_exp(Z, Y, mp)
    return tot


def run(n, M):
    kk = sp.Symbol('k', integer=True)
    def spin(name):
        c = sp.symbols(name + '1:%d' % n)
        return list(c) + [-sum(c)]
    Is, Js, Ks, Ls, Hs = (spin(x) for x in "IJKLH")
    hf = pi / 2
    W = lambda t, A, B: edge_exp(n, M, A, B, t, sp.Rational(1, 2), kk)
    Wb = lambda t, A, B: edge_exp(n, M, A, B, hf - t, 1, kk)
    lhs_star = vertex_exp(n, M, Hs) + Wb(u - v, Ks, Hs) + Wb(up - vp, Js, Hs) + W(up - v, Hs, Is) + W(u - vp, Hs, Ls)
    rhs_star = vertex_exp(n, M, Hs) + Wb(u - v, Hs, Js) + Wb(up - vp, Hs, Ks) + W(up - v, Ls, Hs) + W(u - vp, Is, Hs)
    lhs = sp.expand(W(vp - v, Ls, Ks) + W(up - u, Ls, Js) + lhs_star)
    rhs = sp.expand(W(vp - v, Js, Is) + W(up - u, Ks, Is) + rhs_star)
    hvars = Hs[:-1]
    dL = [sp.simplify(sp.diff(lhs, h)) for h in hvars]
    dR = [sp.simplify(sp.diff(rhs, h)) for h in hvars]
    diff = sp.expand(lhs - rhs)
    ext = [x for S in (Is, Js, Ks, Ls) for x in S[:-1]]
    dE = [sp.simplify(sp.diff(diff, x)) for x in ext]
    return dL, dR, dE


if __name__ == "__main__":
    for n in [2, 3]:
        M = sp.Symbol('M', positive=True)
        dL, dR, dE = run(n, M)
        print("n=%d: d/dh of LHS total:" % n, dL)
        print("      d/dh of RHS total:", dR)
        print("      d/d(external) of LHS-RHS:", dE)
