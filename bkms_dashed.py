"""Dictionary with dashed lines shifted: alpha_{u'}, alpha_{v'} -> + pi/2 (R -> -R), no spin offsets (sum psi = 0).
Spins (psi_h1 = 0): psi_i = -beta_u' + beta_v, psi_j = -beta_u' + beta_v', psi_k = -beta_u + beta_v, psi_l = -beta_u + beta_v';
star 2 saddle psi_h2 = psi_j - beta_u + beta_v.  Check saddle eqs and the full discrete star-star relation."""
import itertools, sys
import numpy as np
from lensnp import PI
from trig_general import TrigGen
from bkms_check import lines

def setup(n, seed, shift=PI / 2):
    rng = np.random.default_rng(seed); al, be = lines(n, rng, "mobius")
    P = [-be["up"] + be["v"], -be["up"] + be["vp"], -be["u"] + be["v"], -be["u"] + be["vp"]]
    a = [al["u"], al["up"] + shift, al["v"], al["vp"] + shift]
    h1 = np.zeros(n, dtype=complex); h2 = P[1] - be["u"] + be["v"]
    return P, a, h1, h2

def saddle_res(m, P, a, h1, h2):
    Pi, Pj, Pk, Pl = P; u, up, v, vp = a; hf = PI / 2
    l1 = [(hf - (u - v), Pk, -1), (hf - (up - vp), Pj, -1), (up - v, Pi, +1), (u - vp, Pl, +1)]
    l2 = [(hf - (u - v), Pj, +1), (hf - (up - vp), Pk, +1), (up - v, Pl, -1), (u - vp, Pi, -1)]
    out = []
    for h, l in [(h1, l1), (h2, l2)]:
        g = m.G(h, l); d = g[:-1] - g[-1]; d = d - 2j * PI * np.round(d.imag / (2 * PI)); out.append(np.max(abs(d)))
    return out

def relation(m, P, a, h1, h2, rng, trials=5):
    n, M = m.n, m.M
    Pi, Pj, Pk, Pl = P; u, up, v, vp = a; hf = PI / 2
    rats = []
    for t in range(trials):
        A = []
        for _ in range(4):
            x = list(rng.integers(0, M, n - 1)) if t else [0] * (n - 1); x.append(-sum(x)); A.append(x)
        AI, AJ, AK, AL = A; L = R = 0
        for ah in itertools.product(range(M), repeat=n - 1):
            ah = list(ah) + [-sum(ah)]
            L += np.exp(m.logS(h1, ah) + m.logW(hf - (u - v), Pk, AK, h1, ah) + m.logW(hf - (up - vp), Pj, AJ, h1, ah)
                        + m.logW(up - v, h1, ah, Pi, AI) + m.logW(u - vp, h1, ah, Pl, AL))
            R += np.exp(m.logS(h2, ah) + m.logW(hf - (u - v), h2, ah, Pj, AJ) + m.logW(hf - (up - vp), h2, ah, Pk, AK)
                        + m.logW(up - v, Pl, AL, h2, ah) + m.logW(u - vp, Pi, AI, h2, ah))
        L *= np.exp(m.logW(vp - v, Pl, AL, Pk, AK) + m.logW(up - u, Pl, AL, Pj, AJ))
        R *= np.exp(m.logW(vp - v, Pj, AJ, Pi, AI) + m.logW(up - u, Pk, AK, Pi, AI))
        rats.append(L / R)
    rats = np.array(rats)
    return np.max(abs(rats / rats[0] - 1))

if __name__ == "__main__":
    for (n, M) in [(2, 3), (2, 5), (3, 2), (3, 3), (4, 2)]:
        for seed in [7, 11]:
            P, a, h1, h2 = setup(n, seed)
            m = TrigGen(n, M)
            r1, r2 = saddle_res(m, P, a, h1, h2)
            print("n=%d M=%d seed=%d : saddle res %.0e %.0e, star-star dev %.1e" % (n, M, seed, r1, r2, relation(m, P, a, h1, h2, np.random.default_rng(0))))
