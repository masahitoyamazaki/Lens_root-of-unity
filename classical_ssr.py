"""(ii) Classical star-star relation at O(1/hbar):
   A^(1)(xi_h1) + Lambda_ext^(1) = A^(2)(xi_h2) + Lambda_ext^(2),
with edge terms sum_ij F(z), F(z) = (i/N) int_0^z log thbar4(N u|tauhat) du (Wbar: z0 = xi_A - xi_B + lam/2 - theta lam/pi),
and vertex term sum_{i<j} C(xi_ij), C(xi) = -(xi~ - pi/(2N))^2 with xi~ reduced to 0<Re N xi~<pi (C even, period pi/N)."""
import sys
import numpy as np
from lensnp import PI
from psi import F
from discrete_ssr import Model

def C(x, N):
    if (N * x).real < 0:
        x = -x
    k = np.floor((N * x).real / PI); x = x - k * PI / N
    return -(x - PI / (2 * N)) ** 2

def run(n, r, N, tau=0.05 + 0.35j, rap=(0.9, 0.5, 0.2, 0.1), seed=0):
    Mo = Model(n, r, N, tau); lam = Mo.lam
    rng = np.random.default_rng(seed)
    def rxi():
        x = rng.uniform(-0.25, 0.25, n) * PI / N; return x - x.mean()
    XI, XJ, XK, XL = [rxi() for _ in range(4)]
    u, up, v, vp = rap; hf = PI / 2
    legs1 = [(hf - (u - v), XK, -1), (hf - (up - vp), XJ, -1), (up - v, XI, +1), (u - vp, XL, +1)]
    legs2 = [(hf - (u - v), XJ, +1), (hf - (up - vp), XK, +1), (up - v, XL, -1), (u - vp, XI, -1)]
    h1, e1 = Mo.saddle(legs1, np.linspace(-0.3, 0.3, n) * PI / N)
    h2, e2 = Mo.saddle(legs2, h1)
    Fz = lambda z: F(z, r, N, tau)
    def W(th, A, B):
        return sum(Fz(A[i] - B[j] + th * lam / PI) for i in range(n) for j in range(n))
    def V(h):
        return sum(C(h[i] - h[j], N) for i in range(n) for j in range(i + 1, n))
    LHS = W(vp - v, XL, XK) + W(up - u, XL, XJ) + V(h1) + W(hf - (u - v), XK, h1) + W(hf - (up - vp), XJ, h1) + W(up - v, h1, XI) + W(u - vp, h1, XL)
    RHS = W(vp - v, XJ, XI) + W(up - u, XK, XI) + V(h2) + W(hf - (u - v), h2, XJ) + W(hf - (up - vp), h2, XK) + W(up - v, XL, h2) + W(u - vp, XI, h2)
    return LHS, RHS, (e1, e2)

if __name__ == "__main__":
    for (n, r, N) in [(2, 1, 3), (2, 2, 2), (3, 1, 2), (3, 2, 2), (3, 1, 3)]:
        for seed in [0, 1]:
            L, R, e = run(n, r, N, seed=seed)
            print("n=%d r=%d N=%d seed=%d : A_LHS = %s  A_RHS = %s  |diff| = %.1e  (saddle res %.0e %.0e)"
                  % (n, r, N, seed, np.round(L, 8), np.round(R, 8), abs(L - R), *e))
