"""General trigonometric limit (towards the BKMS model).
Ansatz: phi_{c,p} = psi_{c,p} + rho_p * pi M T / n,  rho_p = p-(n+1)/2,  psi real, T -> i oo.
For a pair of components (i,j) the shift is c = (i-j)/n and
  log thbar2(x + c pi tau | tau) -> c=0: log(2 cos x) + i pi tau/4
                                    c>0: i pi tau/4 - i pi c tau - i x
                                    c<0: i pi tau/4 + i pi c tau + i x
  log thbar1(x + c pi tau | tau) -> c=0: log(2 sin x) + i pi tau/4
                                    c>0: log(i)  + i pi tau/4 - i pi c tau - i x
                                    c<0: log(-i) + i pi tau/4 + i pi c tau + i x
The divergent constants are kept with a finite symbol tau0 (they must cancel)."""
import itertools, sys
import numpy as np
from lensnp import PI


def lth(j, x, c, tau0):
    if abs(c) < 1e-12:
        f = np.sin(x) if j == 1 else np.cos(x)
        return np.log(complex(2 * f)) + 1j * PI * tau0 / 4
    ph = (np.log(1j) if c > 0 else np.log(-1j)) if j == 1 else 0
    if c > 0:
        return ph + 1j * PI * tau0 / 4 - 1j * PI * c * tau0 - 1j * x
    return ph + 1j * PI * tau0 / 4 + 1j * PI * c * tau0 + 1j * x


class TrigGen:
    def __init__(self, n, M, T0=0.37 + 1.3j, cplx=False):
        self.n, self.M, self.T0, self.cplx = n, M, T0, cplx

    def logw(self, x, c, a):
        M, T0 = self.M, self.T0
        s = a / M * lth(2, x, c, M * T0)
        if a >= 0:
            s -= sum(lth(1, (x + (k + 0.5) * PI) / M, c, T0) for k in range(a))
        else:
            s += sum(lth(1, (x + (k + 0.5) * PI) / M, c, T0) for k in range(a, 0))
        return s

    def logW(self, th, PA, AA, PB, AB):
        n = self.n
        return sum(self.logw(PA[i] - PB[j] + th, (i - j) / n, AA[i] - AB[j]) for i in range(n) for j in range(n))

    def logS(self, P, A):
        n, M, T0 = self.n, self.M, self.T0
        s = 0
        for i in range(n):
            for j in range(i + 1, n):
                aij = A[i] - A[j]
                s += lth(1, (P[i] - P[j] + PI * aij) / M, (i - j) / n, T0) - 1j * PI * aij / M
        return s

    # ---- trigonometric saddle point equation (from saddlemod with log thbar2(.|MT) -> lth(2,...)) ----
    def G(self, ph, legs):
        n, T0, M = self.n, self.T0, self.M
        g = np.zeros(n, dtype=complex)
        for p in range(n):
            for th, X, sg in legs:
                for i in range(n):
                    if sg > 0:
                        g[p] += lth(2, ph[p] - X[i] + th, (p - i) / n, M * T0)
                    else:
                        g[p] -= lth(2, X[i] - ph[p] + th, (i - p) / n, M * T0)
        return g

    def saddle(self, legs, x0, tol=1e-13):
        """real psi_h; equations Re(G_p - G_n) = 0 (p<n); Im(G_p-G_n) must be = 0 mod 2 pi (checked)."""
        n = self.n
        y = np.asarray(x0[:n - 1], dtype=float)
        def full(y):
            return np.concatenate([y, [-np.sum(y)]])
        def F(y):
            g = self.G(full(y), legs)
            return (g[:n - 1] - g[n - 1]).real
        for it in range(200):
            f = F(y)
            if np.max(abs(f)) < tol:
                break
            J = np.zeros((n - 1, n - 1)); h = 1e-7
            for c in range(n - 1):
                dy = np.zeros(n - 1); dy[c] = h
                J[:, c] = (F(y + dy) - f) / h
            step = np.linalg.solve(J, f); t = 1.0
            while t > 1e-4:
                yn = y - t * step
                if np.max(abs(F(yn))) < np.max(abs(f)):
                    break
                t /= 2
            y = yn
        g = self.G(full(y), legs); im = (g - g[-1]).imag
        imres = np.max(abs(im - 2 * PI * np.round(im / (2 * PI))))
        return full(y), max(np.max(abs(F(y))), imres)

    def saddle_c(self, legs, x0, tol=1e-13):
        """complex psi_h; equations G_p - G_n = 2 pi i k_p (k_p fixed by rounding at the start)."""
        n = self.n
        y = np.asarray(x0[:n - 1], dtype=complex)
        full = lambda y: np.concatenate([y, [-np.sum(y)]])
        g0 = self.G(full(y), legs); k = np.round(((g0[:n-1] - g0[n-1]).imag) / (2 * PI))
        def F(y):
            g = self.G(full(y), legs)
            return g[:n - 1] - g[n - 1] - 2j * PI * k
        for it in range(300):
            f = F(y)
            if np.max(abs(f)) < tol:
                break
            J = np.zeros((n - 1, n - 1), dtype=complex); h = 1e-7
            for c in range(n - 1):
                dy = np.zeros(n - 1, dtype=complex); dy[c] = h
                J[:, c] = (F(y + dy) - f) / h
            step = np.linalg.solve(J, f); t = 1.0
            while t > 1e-5:
                yn = y - t * step
                if np.max(abs(F(yn))) < np.max(abs(f)):
                    break
                t /= 2
            y = yn
        return full(y), np.max(abs(F(y)))

    def run(self, rap=(0.9, 0.5, 0.2, 0.1), seed=0, trials=4, spread=0.3):
        n, M = self.n, self.M
        u, up, v, vp = rap; hf = PI / 2
        rng = np.random.default_rng(seed)
        def rpsi():
            x = rng.uniform(-spread, spread, n); return x - x.mean()
        PI_, PJ, PK, PL = [rpsi() for _ in range(4)]
        legs1 = [(hf - (u - v), PK, -1), (hf - (up - vp), PJ, -1), (up - v, PI_, +1), (u - vp, PL, +1)]
        legs2 = [(hf - (u - v), PJ, +1), (hf - (up - vp), PK, +1), (up - v, PL, -1), (u - vp, PI_, -1)]
        sad = self.saddle_c if self.cplx else self.saddle
        h1, e1 = sad(legs1, np.zeros(n))
        h2, e2 = sad(legs2, h1)
        rats = []
        for t in range(trials):
            A = []
            for _ in range(4):
                a = list(rng.integers(0, M, n - 1)) if t else [0] * (n - 1); a.append(-sum(a)); A.append(a)
            AI, AJ, AK, AL = A; L = R = 0
            for ah in itertools.product(range(M), repeat=n - 1):
                ah = list(ah) + [-sum(ah)]
                L += np.exp(self.logS(h1, ah) + self.logW(hf - (u - v), PK, AK, h1, ah) + self.logW(hf - (up - vp), PJ, AJ, h1, ah)
                            + self.logW(up - v, h1, ah, PI_, AI) + self.logW(u - vp, h1, ah, PL, AL))
                R += np.exp(self.logS(h2, ah) + self.logW(hf - (u - v), h2, ah, PJ, AJ) + self.logW(hf - (up - vp), h2, ah, PK, AK)
                            + self.logW(up - v, PL, AL, h2, ah) + self.logW(u - vp, PI_, AI, h2, ah))
            L *= np.exp(self.logW(vp - v, PL, AL, PK, AK) + self.logW(up - u, PL, AL, PJ, AJ))
            R *= np.exp(self.logW(vp - v, PJ, AJ, PI_, AI) + self.logW(up - u, PK, AK, PI_, AI))
            rats.append(L / R)
        rats = np.array(rats)
        return (h1, e1), (h2, e2), np.max(abs(rats / rats[0] - 1))


if __name__ == "__main__":
    for (n, M) in [(2, 3), (2, 4), (3, 2), (3, 3), (4, 2)]:
        for T0 in [0.37 + 1.3j, -0.2 + 0.5j]:
            (h1, e1), (h2, e2), dev = TrigGen(n, M, T0, cplx=True).run()
            print("n=%d M=%d T0=%s  saddle res %.0e %.0e  psi_h1=%s  dev=%.1e"
                  % (n, M, T0, e1, e2, np.round(h1, 4), dev))
