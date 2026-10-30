"""Check: the (r,N,tau) discrete model == (1, rN, tau1) model, tau1 = r tau/(1-(r-1) r N tau),
with spins identified through phi = pi xi/lam  (xi_1 = xi * lam_1/lam) and the same discrete spins a.
(i)  saddle points coincide in phi variables;
(ii) LHS(a)_r / LHS(a)_1 = const * exp((k_r-k_1) n (sum a_l^3 - sum a_i^3)),  k_r = i pi (rN-1)/(3 r^2 N)
     (cubic gauge factor from p(a), Prop. 3.6)."""
import itertools, sys
import numpy as np
from lensnp import PI
from discrete_ssr import Model

def lhs(M, xh, X, A, rap):
    u, up, v, vp = rap; hf = PI/2; n = M.n
    XI, XJ, XK, XL = X; AI, AJ, AK, AL = A
    tot = 0
    for ah in itertools.product(range(M.rN), repeat=n-1):
        ah = list(ah) + [-sum(ah)]
        s = (M.logS(xh, ah) + M.logW(hf-(u-v), XK, AK, xh, ah) + M.logW(hf-(up-vp), XJ, AJ, xh, ah)
             + M.logW(up-v, xh, ah, XI, AI) + M.logW(u-vp, xh, ah, XL, AL))
        tot += np.exp(s)
    return np.exp(M.logW(vp-v, XL, AL, XK, AK) + M.logW(up-u, XL, AL, XJ, AJ)) * tot

def main(n, r, N, tau=0.05+0.35j, rap=(0.9, 0.5, 0.2, 0.1), seed=3):
    Mr = Model(n, r, N, tau)
    tau1 = r*tau/(1-(r-1)*r*N*tau)
    M1 = Model(n, 1, r*N, tau1)
    sc = M1.lam/Mr.lam
    rng = np.random.default_rng(seed)
    def rxi():
        x = rng.uniform(-0.25, 0.25, n)*PI/N; return x - x.mean()
    X = [rxi() for _ in range(4)]; X1 = [sc*x for x in X]
    u, up, v, vp = rap; hf = PI/2
    legs = lambda X: [(hf-(u-v), X[2], -1), (hf-(up-vp), X[1], -1), (up-v, X[0], +1), (u-vp, X[3], +1)]
    x0 = np.linspace(-0.3, 0.3, n)*PI/N
    xh, e = Mr.saddle(legs(X), x0)
    xh1, e1 = M1.saddle(legs(X1), sc*xh)
    print("tau1 =", tau1, " Im tau1 > 0:", tau1.imag > 0)
    print("(i) saddle residuals %.1e %.1e ; |phi_h(r,N) - phi_h(1,rN)| = %.1e" % (e, e1, max(abs(sc*xh - xh1))))
    kr = 1j*PI*(r*N-1)/(3*r*r*N); k1 = 1j*PI*(r*N-1)/(3*r*N)
    vals = []
    for t in range(4):
        A = []
        for _ in range(4):
            a = list(rng.integers(0, r*N, n-1)); a.append(-sum(a)); A.append(a)
        g = np.exp((kr-k1)*n*(sum(np.array(A[3])**3) - sum(np.array(A[0])**3)))
        vals.append(lhs(Mr, xh, X, A, rap) / lhs(M1, xh1, X1, A, rap) / g)
    vals = np.array(vals)
    print("(ii) gauge-corrected LHS_r/LHS_1 :", np.round(vals, 10), " max rel. dev. %.1e" % max(abs(vals/vals[0]-1)))

if __name__ == "__main__":
    main(*(int(s) for s in sys.argv[1:4]))
