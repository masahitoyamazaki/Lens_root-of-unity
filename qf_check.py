"""Identification of the elliptic KM-type model (draft eqs. KMnW, KMnS; general torsion point omega0)
with the A^{(1)}_{n-1} broken Z_N model of Quano-Fujii (hep-th/9210063), using quano_fujii.QF.

 (1) omega0 = pi/n, tau_QF = T, N = M, a0 = 0:  s_i = a_i + i,  u = (theta/pi - 1/2)/M
     W_theta(a,b) = C_theta eps(a)/eps(b) wbar_{s(a) s(b)}(u),  S(a) = const * Delta(s(a))   (eps(a) = +-1)
     and the QF weights (wbar on edges, Delta on sites) satisfy the star-star relation (KMnssr).
 (2) omega0 = pi(MT - nu)/n, gcd(n,M,nu)=1: gauge equivalence with QF at tau_QF = gamma(T),
     gamma = (a b; c d) in SL(2,Z), c = 0 mod M, a = d = 1 mod M, d + nu c/M = 0 mod n, bM + nu a + 1 = 0 mod n,
     u = (theta/pi - 1/2)/(M (cT+d)).   (test: cross-ratio of log W - log QF)
 (3) star-star relation (KMnssr) for omega0 = pi(kappa MT - nu)/n: holds iff omega0 is a primitive n-torsion point
     of pi Z + pi MT Z, i.e. gcd(kappa, nu, n) = 1."""
import itertools, sys
import numpy as np
from lensnp import logthbar, PI
from quano_fujii import QF, wrap

T = 0.13 + 0.55j


def spins_of(n, M):
    return [list(x) + [-sum(x)] for x in itertools.product(range(M), repeat=n - 1)]


def logW(n, M, T, w0, th, a, b):
    s = 0
    for i in range(n):
        for j in range(n):
            d = a[i] - b[j]; x = (i - j) * w0 + th
            f = lambda k: logthbar(1, (x + (k + 0.5) * PI) / M, T)
            s += -sum(f(k) for k in range(d)) if d >= 0 else sum(f(k) for k in range(d, 0))
    return s


def logS(n, M, T, w0, a):
    return sum(logthbar(1, (PI * (a[i] - a[j]) - (j - i) * w0) / M, T) for i in range(n) for j in range(i + 1, n))


def ssr_dev(n, M, lW, lS, seed=11, trials=3):
    """star-star relation (KMnssr); returns max deviation of L/R from a constant, and the constant."""
    spins = spins_of(n, M); rng = np.random.default_rng(seed); Rs = []
    for _ in range(trials):
        u, up, v, vp = rng.uniform(0, 1, 4)
        Wb = lambda t, A, B: lW(PI / 2 - t, A, B)
        i_, j_, k_, l_ = [spins[x] for x in rng.integers(0, len(spins), 4)]
        L = lW(vp - v, l_, k_) + lW(up - u, l_, j_) + np.log(sum(np.exp(lS(h) + Wb(u - v, k_, h) + Wb(up - vp, j_, h)
                                                                    + lW(up - v, h, i_) + lW(u - vp, h, l_)) for h in spins))
        R = lW(vp - v, j_, i_) + lW(up - u, k_, i_) + np.log(sum(np.exp(lS(h) + Wb(u - v, h, j_) + Wb(up - vp, h, k_)
                                                                    + lW(up - v, l_, h) + lW(u - vp, i_, h)) for h in spins))
        Rs.append(L - R)
    return max(abs(wrap(x - Rs[0])) for x in Rs), np.exp(Rs[0])


def qf_map(n, k, A, B):
    sa = [A[i] + k * i for i in range(n)]; sb = [B[i] + k * i for i in range(n)]
    mn = min(sa + sb)
    return [x - mn for x in sa], [x - mn for x in sb]


def part1(n, M):
    w0 = PI / n; qf = QF(n, M, T, np.zeros(n)); spins = spins_of(n, M); Z = [0] * n
    G = lambda th, A, B: qf.log_wbar(*qf_map(n, 1, A, B), (th / PI - 0.5) / M) - logW(n, M, T, w0, th, A, B)
    gauge = 0
    for th in [0.3, 0.8]:
        g0 = G(th, Z, Z)
        for A, B in itertools.product(spins, spins):
            # W/wbar = C_theta eps(a)/eps(b) with eps = +-1:  G(A,B) - g0 = gL(A) + gR(B), gL + gR = 0, exp(2 gL) = 1
            r = G(th, A, B) - g0 - (G(th, A, Z) - g0) - (G(th, Z, B) - g0)
            gauge = max(gauge, abs(wrap(r)), abs(wrap(2 * (G(th, A, Z) - g0))), abs(wrap(G(th, A, Z) + G(th, Z, A) - 2 * g0)))
    dS = [wrap(qf.log_Delta(qf_map(n, 1, A, A)[0]) - logS(n, M, T, w0, A)) for A in spins]
    dS = max(abs(wrap(x - dS[0])) for x in dS)
    cache = {}
    def lWq(t, A, B):
        key = (round(t, 12), tuple(A), tuple(B))
        if key not in cache:
            cache[key] = qf.log_wbar(*qf_map(n, 1, A, B), (t / PI - 0.5) / M)
        return cache[key]
    d, R = ssr_dev(n, M, lWq, lambda h: qf.log_Delta(qf_map(n, 1, h, h)[0]))
    print("(1) n=%d M=%d omega0=pi/n: W/wbar sign gauge dev %.1e ; S/Delta dev %.1e ; SSR with QF (wbar, Delta) dev %.1e"
          % (n, M, gauge, dS, d))


def gamma_rule(n, M, nu):
    for c in [M, -M, 2 * M, 3 * M]:
        for d in range(-15, 16):
            if np.gcd(c, d) != 1 or (d - 1) % M or (d + nu * c // M) % n:
                continue
            for a in range(-40, 41):
                if (a * d - 1) % c or (a - 1) % M:
                    continue
                b = (a * d - 1) // c
                if (b * M + nu * a + 1) % n == 0:
                    return a, b, c, d
    return None


def part2(n, M, nu):
    g = gamma_rule(n, M, nu)
    if g is None:
        print("(2) n=%d M=%d nu=%d: no gamma (gcd(n,M,nu)=%d)" % (n, M, nu, np.gcd.reduce([n, M, nu]))); return
    a_, b_, c_, d_ = g; J = c_ * T + d_; tq = (a_ * T + b_) / J
    qf = QF(n, M, tq, np.zeros(n)); w0 = PI * (M * T - nu) / n; th = 0.37; spins = spins_of(n, M)
    best = np.inf
    for k in range(M):
        F = lambda A, B: qf.log_Wbar(*qf_map(n, k, A, B), (th / PI - 0.5) / (M * J)) - logW(n, M, T, w0, th, A, B)
        rng = np.random.default_rng(5); dev = 0
        for _ in range(5):
            A, Ap, B, Bp = [spins[x] for x in rng.integers(0, len(spins), 4)]
            dev = max(dev, abs(wrap(F(A, B) + F(Ap, Bp) - F(A, Bp) - F(Ap, B))))
        best = min(best, dev)
    print("(2) n=%d M=%d nu=%d gamma=%s: cross-ratio dev %.1e" % (n, M, nu, g, best))


def part3(n, M, kap, nu):
    w0 = PI * (kap * M * T - nu) / n
    d, R = ssr_dev(n, M, lambda t, A, B: logW(n, M, T, w0, t, A, B), lambda A: logS(n, M, T, w0, A))
    print("(3) n=%d M=%d omega0=pi(%d MT - %d)/n  gcd(kappa,nu,n)=%d : SSR dev %.1e  R=%s"
          % (n, M, kap, nu, np.gcd.reduce([kap, nu, n]), d, np.round(R, 8)))


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    for n, M in [(2, 3), (3, 2), (3, 3), (4, 2)]:
        part1(n, M)
    for n, M in [(2, 3), (3, 2), (3, 4), (4, 3), (3, 3), (4, 2)]:
        for nu in range(n):
            part2(n, M, nu)
    for n, M, kap, nu in [(3, 2, 0, 1), (3, 2, 2, 1), (3, 3, 0, 2), (3, 3, -1, 0), (4, 2, -1, 0), (4, 2, 2, 1),
                          (4, 2, 0, 2), (4, 2, 2, 0), (4, 2, 2, 2), (3, 3, 0, 0)]:
        part3(n, M, kap, nu)
