"""Reality and positivity of the elliptic Kashiwara-Miwa type weights (draft Sec. 'Reality and positivity').
T = i t (real nome), real rapidities, omega0 = pi(MT - nu)/n.
(1) conjugation: W_theta(a,b)^* = W_theta(-b,-a) and S(a)^* = +-S(-a) for nu = 0 (omega0 imaginary);
(2) IRF weight V(I,J,K,L) = W(L,K)W(L,J) sum_h S(h) Wb(K,h) Wb(J,h) W(h,I) W(h,L):  V^* = V(-L,-K,-J,-I);
(3) positivity of edge weights / IRF weights after the gauge W -> W * e^{f(a)-f(b)}."""
import itertools, numpy as np, warnings
warnings.filterwarnings("ignore")
from lensnp import PI
from qf_check import logW, logS, spins_of


def setup(n, M, t, nu):
    T = 1j * t
    w0 = PI * (M * T - nu) / n
    W = lambda th, a, b: np.exp(logW(n, M, T, w0, th, a, b))
    S = lambda a: np.exp(logS(n, M, T, w0, a))
    return W, S


def irf(W, S, spins, rap, I, J, K, L):
    u, up, v, vp = rap; hf = PI / 2
    Wb = lambda th, a, b: W(hf - th, a, b)
    return W(vp - v, L, K) * W(up - u, L, J) * sum(S(h) * Wb(u - v, K, h) * Wb(up - vp, J, h) * W(up - v, h, I) * W(u - vp, h, L) for h in spins)


if __name__ == "__main__":
    neg = lambda a: [-x for x in a]
    for n, M in [(2, 3), (3, 2), (3, 3), (4, 2)]:
        for nu in [0, 1]:
            W, S = setup(n, M, 0.55, nu)
            spins = spins_of(n, M); rng = np.random.default_rng(0)
            th = 0.4
            cW = max(abs(np.conj(W(th, a, b)) - W(th, neg(b), neg(a))) / abs(W(th, a, b)) for a in spins for b in spins)
            cS = max(min(abs(np.conj(S(a)) - S(neg(a))), abs(np.conj(S(a)) + S(neg(a)))) / abs(S(a)) for a in spins)
            rap = (0.9, 0.5, 0.2, 0.1)
            dv = 0; Vals = []
            for _ in range(5):
                I, J, K, L = [spins[x] for x in rng.integers(0, len(spins), 4)]
                V = irf(W, S, spins, rap, I, J, K, L); V2 = irf(W, S, spins, rap, neg(L), neg(K), neg(J), neg(I))
                dv = max(dv, min(abs(np.conj(V) - V2), abs(np.conj(V) + V2)) / abs(V)); Vals.append(V)
            argW = [np.angle(W(th, a, b)) for a in spins for b in spins]
            print("n=%d M=%d nu=%d: W*=W(-b,-a) dev %.1e ; S*=+-S(-a) dev %.1e ; V*=+-V(-L,-K,-J,-I) dev %.1e ; arg W range [%.2f, %.2f]"
                  % (n, M, nu, cW, cS, dv, min(argW), max(argW)))
