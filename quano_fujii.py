"""Compare the elliptic Z_M^{n-1} Kashiwara-Miwa type model (draft, eqs. KMnW, KMnS) with the
A^{(1)}_{n-1} broken Z_N model of Quano-Fujii (hep-th/9210063, Theorem 4.4).

QF:  h(z) ~ theta_1(pi z | tau_QF);  abar_j = a_j + rho_j + (2j+1-n)/(2n)  (a_j = (a|eps_j));
     wbar_{a+eps_i, b+eps_k}/wbar_{ab} =
        prod_{m<i} h(u+(abar_m-bbar_k)w) prod_{m>i} h(-u-(abar_m-bbar_k)w)
      / [ prod_{m<k} h(-u-(abar_i-bbar_m+1)w) prod_{m>k} h(u+(abar_i-bbar_m+1)w) ],   w = 1/N,
     Wbar_ab = sqrt(Delta(a) Delta(b)) wbar_ab,  Delta(a) = prod_{l<m} h(a_lm w),  a_lm = abar_l - abar_m.
A step a -> a + eps_bar_i changes (a|eps_m) by delta_mi - 1/n.
Test: cross-ratio of log(QF) - log(ours) as functions of the two spins (gauge equivalence f(a)g(b))."""
import itertools
import numpy as np
from lensnp import logthbar, PI


def logh(z, tq):
    return logthbar(1, PI * z, tq)


class QF:
    def __init__(self, n, N, tq, a0):
        self.n, self.N, self.tq = n, N, tq
        self.w = 1.0 / N
        self.a0 = np.asarray(a0, dtype=complex)   # continuous background (a0|eps_j)
        self.rho = np.array([n - 1 - j for j in range(n)], dtype=float)  # (rho|eps_j) up to a constant

    def abar(self, steps):
        """steps: integer vector s (number of eps_bar_i steps from a0). returns abar_j."""
        n = self.n
        s = np.asarray(steps)
        a = self.a0 + s - s.sum() / n
        return a + self.rho + (2 * np.arange(n) + 1 - n) / (2 * n)

    def log_wbar_step(self, sa, sb, i, k, u):
        n, w, tq = self.n, self.w, self.tq
        A = self.abar(sa); B = self.abar(sb)
        num = sum(logh(u + (A[m] - B[k]) * w, tq) for m in range(i)) + \
              sum(logh(-u - (A[m] - B[k]) * w, tq) for m in range(i + 1, n))
        den = sum(logh(-u - (A[i] - B[m] + 1) * w, tq) for m in range(k)) + \
              sum(logh(u + (A[i] - B[m] + 1) * w, tq) for m in range(k + 1, n))
        return num - den

    def log_wbar(self, sa, sb, u):
        """walk from (0,0) to (sa,sb) by simultaneous steps; requires sum(sa) == sum(sb) (pad with full cycles)."""
        n = self.n
        sa = list(sa); sb = list(sb)
        da, db = sum(sa), sum(sb)
        # use the N-periodicity a ~ a + N eps_bar_0 to make the step numbers congruent mod n; pairs of states in
        # different sectors (step numbers differing mod gcd(n,N)) are not connected by the recursion: weight 0
        for _ in range(n):
            if (da - db) % n == 0:
                break
            sb[0] += self.N; db += self.N
        else:
            return -np.inf
        # make total step numbers equal by adding full cycles (sum eps_bar = 0, i.e. one step in every direction)
        while da < db:
            sa = [x + 1 for x in sa]; da += n
        while db < da:
            sb = [x + 1 for x in sb]; db += n
        while da < db:
            sa = [x + 1 for x in sa]; da += n
        assert da == db
        la = [i for i in range(n) for _ in range(sa[i])]
        lb = [k for k in range(n) for _ in range(sb[k])]
        ca = [0] * n; cb = [0] * n; val = 0
        for i, k in zip(la, lb):
            val += self.log_wbar_step(ca, cb, i, k, u)
            ca[i] += 1; cb[k] += 1
        return val

    def log_Delta(self, s):
        A = self.abar(s); n = self.n
        return sum(logh((A[l] - A[m]) * self.w, self.tq) for l in range(n) for m in range(l + 1, n))

    def log_Wbar(self, sa, sb, u):
        return 0.5 * (self.log_Delta(sa) + self.log_Delta(sb)) + self.log_wbar(sa, sb, u)


class Ours:
    """elliptic KM-type weight (KMnW): prod_{i,j} prod_{k<a_i-b_j} thbar1(((i-j)w0+theta+(k+1/2)pi)/M | T)^{-1}"""
    def __init__(self, n, M, T, nu):
        self.n, self.M, self.T = n, M, T
        self.w0 = PI * (M * T - nu) / n

    def logW(self, th, a, b):
        n, M, T = self.n, self.M, self.T
        s = 0
        for i in range(n):
            for j in range(n):
                d = a[i] - b[j]; x = (i - j) * self.w0 + th
                f = lambda k: logthbar(1, (x + (k + 0.5) * PI) / M, T)
                s += -sum(f(k) for k in range(d)) if d >= 0 else sum(f(k) for k in range(d, 0))
        return s


def wrap(x):
    return x - 2j * PI * np.round(x.imag / (2 * PI))


def cross_dev(F, n, M, rng, trials=6):
    dev = 0
    for _ in range(trials):
        sp = []
        for _ in range(4):
            x = list(rng.integers(0, M, n - 1)); x.append(-sum(x)); sp.append(x)
        a, ap, b, bp = sp
        cr = F(a, b) + F(ap, bp) - F(a, bp) - F(ap, b)
        dev = max(dev, abs(wrap(cr)))
    return dev


if __name__ == "__main__":
    n, M = 3, 3
    T = 0.1 + 0.6j
    rng = np.random.default_rng(0)
    results = []
    for nu in [0, 1]:
        ours = Ours(n, M, T, nu)
        for tq_name, tq in [("T", T), ("T/n", T / n), ("nT", n * T)]:
            for a0name, a0 in [("0", np.zeros(n)), ("rhoT", np.array([j * T for j in range(n)])),
                               ("-rhoT", np.array([-j * T for j in range(n)])), ("rhoT/n", np.array([j * T / n for j in range(n)]))]:
                qf = QF(n, M, tq, a0)
                for uscale in [1, -1]:
                    for uoff in [0, 0.5, -0.5, 0.25]:
                        th = 0.37
                        u = uscale * (th / PI) / M + uoff / M
                        # spins: our a (sum 0) <-> QF steps s = a (+ const)
                        def F(a, b):
                            sa = [x - min(a) for x in a]; sb = [x - min(b) for x in b]
                            return qf.log_Wbar(sa, sb, u) - ours.logW(th, a, b)
                        try:
                            d = cross_dev(F, n, M, np.random.default_rng(1))
                        except Exception as e:
                            d = np.inf
                        results.append((d, nu, tq_name, a0name, uscale, uoff))
    results.sort(key=lambda r: r[0])
    for r in results[:12]:
        print("dev %.1e  nu=%d tau_QF=%s a0=%s u=%+d*theta/(pi M)%+.2f/M" % r)
