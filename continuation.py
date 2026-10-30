"""Continuation from a verified configuration to a target configuration, tracking
 (a) the saddle points h1, h2 (Newton on the branch-free equations exp(G_p - G_n) = 1),
 (b) the branches of log(2 cos x) of every (edge, component) argument (unwrapping),
then evaluating the discrete star-star relation with the tracked branches."""
import itertools
import numpy as np
from lensnp import PI
from trig_general import TrigGen, lth


def edges(P, h1, h2, al):
    """list of (key, kind, theta, A, B) for all 12 edges; A, B are spin arrays."""
    Pi, Pj, Pk, Pl = P
    u, up, v, vp = al
    hf = PI / 2
    return [("Elk", vp - v, Pl, Pk), ("Elj", up - u, Pl, Pj),
            ("1kh", hf - (u - v), Pk, h1), ("1jh", hf - (up - vp), Pj, h1), ("1hi", up - v, h1, Pi), ("1hl", u - vp, h1, Pl),
            ("Eji", vp - v, Pj, Pi), ("Eki", up - u, Pk, Pi),
            ("2hj", hf - (u - v), h2, Pj), ("2hk", hf - (up - vp), h2, Pk), ("2lh", up - v, Pl, h2), ("2ih", u - vp, Pi, h2)]


class Cont(TrigGen):
    def __init__(self, n, M, T0=0.37 + 1.3j):
        TrigGen.__init__(self, n, M, T0)
        self.L = {}   # tracked log(2cos x) per (edge key, component)

    def legs(self, P, al):
        Pi, Pj, Pk, Pl = P; u, up, v, vp = al; hf = PI / 2
        l1 = [(hf - (u - v), Pk, -1), (hf - (up - vp), Pj, -1), (up - v, Pi, +1), (u - vp, Pl, +1)]
        l2 = [(hf - (u - v), Pj, +1), (hf - (up - vp), Pk, +1), (up - v, Pl, -1), (u - vp, Pi, -1)]
        return l1, l2

    def newton(self, legs, h, tol=1e-13):
        n = self.n
        y = np.asarray(h[:n - 1], dtype=complex)
        full = lambda y: np.concatenate([y, [-np.sum(y)]])
        def F(y):
            g = self.G(full(y), legs)
            return np.exp(g[:n - 1] - g[n - 1]) - 1
        for it in range(100):
            f = F(y)
            if np.max(abs(f)) < tol:
                break
            J = np.zeros((n - 1, n - 1), dtype=complex); hh = 1e-7
            for c in range(n - 1):
                dy = np.zeros(n - 1, dtype=complex); dy[c] = hh
                J[:, c] = (F(y + dy) - f) / hh
            step = np.linalg.lstsq(J, f, rcond=None)[0]
            t = 1.0
            while t > 1e-6:
                yn = y - t * step
                if np.max(abs(F(yn))) < np.max(abs(f)):
                    break
                t /= 2
            y = yn
        return full(y), np.max(abs(F(y)))

    def track_logs(self, P, h1, h2, al):
        for key, th, A, B in edges(P, h1, h2, al):
            for c in range(self.n):
                x = A[c] - B[c] + th
                val = np.log(complex(2 * np.cos(x)))
                if (key, c) in self.L:
                    prev = self.L[(key, c)]
                    val = val + 2j * PI * np.round((prev - val).imag / (2 * PI))
                self.L[(key, c)] = val

    def logw_key(self, key, th, A, B, AA, AB):
        """edge weight with tracked fractional-power branches (diagonal part); off-diagonal from lth."""
        n, M, T0 = self.n, self.M, self.T0
        s = 0
        for i in range(n):
            for j in range(n):
                x = A[i] - B[j] + th; c = (i - j) / n; a = AA[i] - AB[j]
                if i == j:
                    s += a / M * (self.L[(key, i)] + 1j * PI * M * T0 / 4)
                else:
                    s += a / M * lth(2, x, c, M * T0)
                if a >= 0:
                    s -= sum(lth(1, (x + (k + 0.5) * PI) / M, c, T0) for k in range(a))
                else:
                    s += sum(lth(1, (x + (k + 0.5) * PI) / M, c, T0) for k in range(a, 0))
        return s

    def relation(self, P, h1, h2, al, rng, trials=5):
        n, M = self.n, self.M
        E = {e[0]: e for e in edges(P, h1, h2, al)}
        W = lambda key, AA, AB: self.logw_key(key, E[key][1], E[key][2], E[key][3], AA, AB)
        rats = []
        for t in range(trials):
            A = []
            for _ in range(4):
                a = list(rng.integers(0, M, n - 1)) if t else [0] * (n - 1); a.append(-sum(a)); A.append(a)
            AI, AJ, AK, AL = A; Lsum = Rsum = 0
            for ah in itertools.product(range(M), repeat=n - 1):
                ah = list(ah) + [-sum(ah)]
                Lsum += np.exp(self.logS(h1, ah) + W("1kh", AK, ah) + W("1jh", AJ, ah) + W("1hi", ah, AI) + W("1hl", ah, AL))
                Rsum += np.exp(self.logS(h2, ah) + W("2hj", ah, AJ) + W("2hk", ah, AK) + W("2lh", AL, ah) + W("2ih", AI, ah))
            Lsum *= np.exp(W("Elk", AL, AK) + W("Elj", AL, AJ))
            Rsum *= np.exp(W("Eji", AJ, AI) + W("Eki", AK, AI))
            rats.append(Lsum / Rsum)
        rats = np.array(rats)
        return np.max(abs(rats / rats[0] - 1))

    def continue_to(self, P0, al0, P1, al1, steps=400, h10=None, h20=None, check_every=None, rng=None, bump=None):
        n = self.n
        l1, l2 = self.legs(P0, al0)
        h1 = h10 if h10 is not None else self.saddle_c(l1, np.zeros(n))[0]
        h2 = h20 if h20 is not None else self.saddle_c(l2, h1)[0]
        self.L = {}
        self.track_logs(P0, h1, h2, al0)
        out = []
        for s in range(1, steps + 1):
            t = s / steps
            P = [(1 - t) * a + t * b for a, b in zip(P0, P1)]
            if bump is not None:
                P = [p + np.sin(PI * t) * d for p, d in zip(P, bump)]
            al = [(1 - t) * a + t * b for a, b in zip(al0, al1)]
            l1, l2 = self.legs(P, al)
            h1, e1 = self.newton(l1, h1); h2, e2 = self.newton(l2, h2)
            self.track_logs(P, h1, h2, al)
            if check_every and (s % check_every == 0 or s == steps):
                out.append((t, e1, e2, self.relation(P, h1, h2, al, rng)))
        return h1, h2, out
