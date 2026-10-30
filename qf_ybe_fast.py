"""Vectorised version of qf_ybe.py for larger state spaces (e.g. n=4, N=3: 27 states).
YBE (QFYBE) checked on slices a=(a1,a2,a3) fixed, all d, via einsum; (SSRQF) and (KMnssr) as in qf_ybe.py."""
import sys, numpy as np, warnings, time
warnings.filterwarnings("ignore")
from qf_ybe import QFS, kmn_vs_qf


def box(M, z1, z1p, z2, z2p):
    A, B, C, D = M.W(z1, z2), M.W(z2, z1p), M.W(z1p, z2p), M.W(z1, z2p)
    # T[a, dd, b, c] = A[a,b] B[b,c] C[c,dd] / D[a,dd]
    return np.einsum('ab,bc,cd,ad->adbc', A, B, C, 1 / D)


def ybe_slices(M, z, nslices=6, seed=0):
    z1, z1p, z2, z2p, z3, z3p = z
    A = box(M, z1, z1p, z2, z2p)   # S(1;2)
    B = box(M, z1, z1p, z3, z3p)   # S(1;3)
    C = box(M, z2, z2p, z3, z3p)   # S(2;3)
    d = M.d; rng = np.random.default_rng(seed); dev = 0
    for _ in range(nslices):
        a1, a2, a3 = rng.integers(0, d, 3)
        # LHS = S23(1;2) S12(1;3) S23(2;3): L[d1,d2,d3] = sum_{b2,b3,c2} A[a2,a3,b2,b3] B[a1,b2,d1,c2] C[c2,b3,d2,d3]
        L = np.einsum('xy,xiv,vyjk->ijk', A[a2, a3], B[a1], C, optimize=True)
        # RHS = S12(2;3) S23(1;3) S12(1;2): R[d1,d2,d3] = sum_{b1,b2,c2} C[a1,a2,b1,b2] B[b2,a3,c2,d3] A[b1,c2,d1,d2]
        R = np.einsum('pq,qrk,prij->ijk', C[a1, a2], B[:, a3], A, optimize=True)
        dev = max(dev, np.max(abs(L - R)) / np.max(abs(L)))
    return dev


def ssr_qf_fast(M, z1, z1p, z2, z2p):
    A, B, C, D, E, F = M.W(z1, z2), M.W(z2, z1p), M.W(z1p, z2p), M.W(z1, z2p), M.W(z1, z1p), M.W(z2, z2p)
    iD = 1 / D
    # s1[I,J,K,L] = sum_H A[J,H] B[H,I] C[K,H] / D[L,H] ; s2[I,J,K,L] = sum_H C[H,J] A[H,K] B[L,H] / D[H,I]
    s1 = np.einsum('jh,hi,kh,lh->ijkl', A, B, C, iD, optimize=True)
    s2 = np.einsum('hj,hk,lh,hi->ijkl', C, A, B, iD, optimize=True)
    lhs = np.einsum('lk,lj,ijkl->ijkl', E, F, s1, optimize=True)
    rhs = np.einsum('ji,ki,ijkl->ijkl', E, F, s2, optimize=True)
    return np.max(abs(lhs - rhs) / abs(lhs))


if __name__ == "__main__":
    T = 0.13 + 0.55j
    n, N = (int(x) for x in sys.argv[1:3])
    t0 = time.time(); M = QFS(n, N, T)
    z = np.array([0.31, -0.12, 0.07, 0.44, -0.23, 0.19]) * 0.7
    print("n=%d N=%d (%d states): YBE (generic, %d slices) rel. dev %.1e  [%.0fs]" % (n, N, M.d, 6, ybe_slices(M, z), time.time() - t0), flush=True)
    z1, z1p, z2, z2p = z[:4]
    print("   (SSRQF): %.1e ; (KMnssr) same lines: %.1e  [%.0fs]" % (ssr_qf_fast(M, z1, z1p, z2, z2p), kmn_vs_qf(n, N, T, z1, z1p, z2, z2p), time.time() - t0), flush=True)
