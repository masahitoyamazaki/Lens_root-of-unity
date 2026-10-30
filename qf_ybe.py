"""Quano-Fujii S-matrix S(z1,z1';z2,z2') (hep-th/9210063, Sec. 4.1), its Yang-Baxter equation (Thm 4.6),
and the reduction z3 = z1', z3' = z2' to the star-star relation (draft Sec. 5.6):
  <a,d|S|b,c> = Wbar_ab(z1,z2) Wbar_bc(z2,z1') Wbar_cd(z1',z2') / Wbar_ad(z1,z2'),   Wbar_ab(z,z') = Wbar^QF_ab((z-z')/n),
  YBE: S23(1;2) S12(1;3) S23(2;3) = S12(2;3) S23(1;3) S12(1;2).
With z3=z1', z3'=z2' and Wbar(0)=delta the YBE becomes
  E_LK F_LJ sum_h A_Jh B_hI C_Kh / D_Lh = E_JI F_KI sum_h C_hJ A_hK B_Lh / D_hI,
A=Wbar(z1,z2), B=Wbar(z2,z1'), C=Wbar(z1',z2'), D=Wbar(z1,z2'), E=Wbar(z1,z1'), F=Wbar(z2,z2'),
which is (KMnssr) in QF normalisation with (z1,z1')=(zeta_v',zeta_v), (z2,z2')=(zeta_u',zeta_u)."""
import itertools, numpy as np, warnings
warnings.filterwarnings("ignore")
from lensnp import PI
from quano_fujii import QF, wrap
from qf_check import logW, logS, spins_of


def states(n, N):
    """lifts to Z^n of the N^{n-1} classes of Z^n/(Z(1,..,1)+N Z^n), all with the same step sum (needs gcd(n,N)=1),
    so that every pair is connected by the recursion without using the N-periodicity."""
    assert np.gcd(n, N) == 1
    lifts = []
    for x in itertools.product(range(N), repeat=n - 1):
        v = [0] + list(x)
        while sum(v) % n:
            v[0] += N
        lifts.append(v)
    top = max(sum(v) for v in lifts)
    return [tuple(c + (top - sum(v)) // n for c in v) for v in lifts]


def wbar_matrix(qf, st, u):
    d = len(st)
    Wm = np.zeros((d, d), dtype=complex)
    for i, a in enumerate(st):
        for j, b in enumerate(st):
            Wm[i, j] = np.exp(qf.log_Wbar(list(a), list(b), u))
    return Wm


class QFS:
    def __init__(self, n, N, tq):
        self.n, self.N, self.qf = n, N, QF(n, N, tq, np.zeros(n))
        self.st = states(n, N); self.d = len(self.st); self.cache = {}

    def W(self, z, zp):
        key = round(z - zp, 12)
        if key not in self.cache:
            self.cache[key] = wbar_matrix(self.qf, self.st, (z - zp) / self.n)
        return self.cache[key]

    def S(self, z1, z1p, z2, z2p):
        """matrix on H (x) H, <a,d|S|b,c>, index (a,d) -> a*d_ + d."""
        A, B, C, D = self.W(z1, z2), self.W(z2, z1p), self.W(z1p, z2p), self.W(z1, z2p)
        d = self.d
        S = np.zeros((d * d, d * d), dtype=complex)
        for a in range(d):
            for b in range(d):
                for c in range(d):
                    for dd in range(d):
                        S[a * d + dd, b * d + c] = A[a, b] * B[b, c] * C[c, dd] / D[a, dd] if D[a, dd] != 0 else 0
        return S


def ybe_dev(M, z1, z1p, z2, z2p, z3, z3p):
    d = M.d; I = np.eye(d)
    S12 = lambda *z: np.kron(M.S(*z), I); S23 = lambda *z: np.kron(I, M.S(*z))
    L = S23(z1, z1p, z2, z2p) @ S12(z1, z1p, z3, z3p) @ S23(z2, z2p, z3, z3p)
    R = S12(z2, z2p, z3, z3p) @ S23(z1, z1p, z3, z3p) @ S12(z1, z1p, z2, z2p)
    return np.max(abs(L - R)) / np.max(abs(L))


def ssr_qf_dev(M, z1, z1p, z2, z2p):
    """(SSR') directly in terms of the QF matrices."""
    A, B, C, D, E, F = M.W(z1, z2), M.W(z2, z1p), M.W(z1p, z2p), M.W(z1, z2p), M.W(z1, z1p), M.W(z2, z2p)
    d = M.d; dev = 0
    for I, J, K, L in itertools.product(range(d), repeat=4):
        s1 = sum(A[J, h] * B[h, I] * C[K, h] / D[L, h] for h in range(d) if D[L, h] != 0)
        s2 = sum(C[h, J] * A[h, K] * B[L, h] / D[h, I] for h in range(d) if D[h, I] != 0)
        lhs, rhs = E[L, K] * F[L, J] * s1, E[J, I] * F[K, I] * s2
        if lhs != 0:
            dev = max(dev, abs(lhs - rhs) / abs(lhs))
    return dev


def kmn_vs_qf(n, M_, T, z1, z1p, z2, z2p):
    """compare the QF form (SSR') with our (KMnssr) at omega0 = pi/n via zeta_v = n v/(pi M) etc."""
    v, u = PI * M_ * z1p / n, PI * M_ * z2p / n
    vp, up = PI * M_ * z1 / n + PI / 2, PI * M_ * z2 / n + PI / 2
    w0 = PI / n; spins = spins_of(n, M_)
    lW = lambda t, a, b: logW(n, M_, T, w0, t, a, b)
    lS = lambda a: logS(n, M_, T, w0, a)
    Wb = lambda t, a, b: lW(PI / 2 - t, a, b)
    rng = np.random.default_rng(0); Rs = []
    for _ in range(3):
        i_, j_, k_, l_ = [spins[x] for x in rng.integers(0, len(spins), 4)]
        Lh = lW(vp - v, l_, k_) + lW(up - u, l_, j_) + np.log(sum(np.exp(lS(h) + Wb(u - v, k_, h) + Wb(up - vp, j_, h)
                                                                    + lW(up - v, h, i_) + lW(u - vp, h, l_)) for h in spins))
        Rh = lW(vp - v, j_, i_) + lW(up - u, k_, i_) + np.log(sum(np.exp(lS(h) + Wb(u - v, h, j_) + Wb(up - vp, h, k_)
                                                                    + lW(up - v, l_, h) + lW(u - vp, i_, h)) for h in spins))
        Rs.append(Lh - Rh)
    return max(abs(wrap(x - Rs[0])) for x in Rs)


if __name__ == "__main__":
    T = 0.13 + 0.55j
    for n, N in [(2, 3), (3, 2), (2, 5), (3, 4), (4, 3)]:
        M = QFS(n, N, T)
        z = np.array([0.31, -0.12, 0.07, 0.44, -0.23, 0.19]) * 0.7
        print("n=%d N=%d: YBE (generic 6 parameters) rel. dev %.1e" % (n, N, ybe_dev(M, *z)), flush=True)
        z1, z1p, z2, z2p = z[:4]
        print("          YBE at z3=z1', z3'=z2': %.1e ; (SSR') in QF form: %.1e ; (KMnssr) at omega0=pi/n, same lines: %.1e"
              % (ybe_dev(M, z1, z1p, z2, z2p, z1p, z2p), ssr_qf_dev(M, z1, z1p, z2, z2p), kmn_vs_qf(n, N, T, z1, z1p, z2, z2p)), flush=True)


def reduction_check(M, z1, z1p, z2, z2p, eps=1e-6, trials=6, seed=1):
    """YBE at z3=z1'+eps, z3'=z2'+eps versus the factorised form obtained with Wbar(0)=delta (draft Sec. 5.6)."""
    d = M.d; I = np.eye(d)
    S12 = lambda *z: np.kron(M.S(*z), I); S23 = lambda *z: np.kron(I, M.S(*z))
    z3, z3p = z1p + eps, z2p + eps
    Lm = S23(z1, z1p, z2, z2p) @ S12(z1, z1p, z3, z3p) @ S23(z2, z2p, z3, z3p)
    Rm = S12(z2, z2p, z3, z3p) @ S23(z1, z1p, z3, z3p) @ S12(z1, z1p, z2, z2p)
    A, B, C, D, E, F = M.W(z1, z2), M.W(z2, z1p), M.W(z1p, z2p), M.W(z1, z2p), M.W(z1, z1p), M.W(z2, z2p)
    c0 = M.W(z1p + eps, z1p)[0, 0]          # Wbar(0) = c0 * identity (c0 independent of the state)
    rng = np.random.default_rng(seed); dev = 0
    for _ in range(trials):
        a1, a2, a3, d1, d2, d3 = rng.integers(0, d, 6)
        I_, J_, K_, L_ = d3, a2, d1, a1
        s1 = sum(A[J_, h] * B[h, I_] * C[K_, h] / D[L_, h] for h in range(d))
        s2 = sum(C[h, J_] * A[h, K_] * B[L_, h] / D[h, I_] for h in range(d))
        common = (C[I_, a3] / D[J_, a3]) * B[K_, d2] * C[d2, I_]
        predL = c0 ** 2 * E[L_, K_] * common / F[K_, I_] * s1
        predR = c0 ** 2 * E[J_, I_] * common / F[L_, J_] * s2
        row, col = (a1 * d + a2) * d + a3, (d1 * d + d2) * d + d3
        dev = max(dev, abs(Lm[row, col] / predL - 1), abs(Rm[row, col] / predR - 1))
    return dev


if __name__ == "__main__":
    for n, N in [(2, 3), (3, 2), (3, 4)]:  # reduction check
        M = QFS(n, N, T)
        z1, z1p, z2, z2p = np.array([0.31, -0.12, 0.07, 0.44]) * 0.7
        print("n=%d N=%d: YBE sides at z3=z1'+eps, z3'=z2'+eps vs factorised form (eps=1e-6): rel. dev %.1e"
              % (n, N, reduction_check(M, z1, z1p, z2, z2p)), flush=True)
