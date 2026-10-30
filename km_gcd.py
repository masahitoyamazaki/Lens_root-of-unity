# used for draft Sec. "Star-star relation from the Yang-Baxter equation", paragraph "The case gcd(n,M)>1"
"""gcd(n,M)>1 at the QF point omega0=pi/n:
 (1) the kernel generator a_ker=(M/g)(1-n,1,...,1) of Q/MQ -> Pbar/M Pbar changes W_theta(a,b) by a factor which depends on
     (a,b) and is not a gauge factor f(a)f'(b) (cross-ratio test); for (2,2) the ratio is +-rho^{+-1},
     rho = thbar1(theta/2-pi/4|T)/thbar1(theta/2+pi/4|T), exponent +1 for a=b, -1 for a!=b.
 (2) cyclic shift C:(a_1,..,a_n)->(a_2,..,a_n,a_1-1): W(Ca,Cb)=W(a,b), S(Ca)=(-1)^(n-1) S(a).
 (3) box weight W_t1(a,b)W_t2(b,c)W_t3(c,d)/W_t4(a,d) is invariant under a change of lift a->a+M(e_i-e_j) in any slot."""
import numpy as np
from lensnp import PI, logthbar
from quano_fujii import Ours, wrap
T = 0.13 + 0.55j

def logS(n, M, T, w0, a):
    return sum(logthbar(1, (PI*(a[i]-a[j]) - (j-i)*w0)/M, T) for i in range(n) for j in range(i+1, n))

def rnd(rng, n, M):
    a = list(rng.integers(0, M, n-1)); a.append(-sum(a)); return a

if __name__ == '__main__':
    o = Ours(2, 2, T, 0); o.w0 = PI/2
    for th in [0.37, 1.1]:
        rho = logthbar(1, th/2 - PI/4, T) - logthbar(1, th/2 + PI/4, T)
        for a in [0, 1]:
            for b in [0, 1]:
                r = o.logW(th, [a-1, 1-a], [b, -b]) - o.logW(th, [a, -a], [b, -b])
                print(f"(2,2) th={th} a={a} b={b}: ratio/(rho^(+-1)) = exp({wrap(r - (rho if a == b else -rho)):.3f})  (expect 0 or i pi)")
    for n, M in [(2, 2), (2, 4), (3, 3), (3, 6), (4, 2), (4, 4)]:
        o = Ours(n, M, T, 0); o.w0 = PI/n; g = int(np.gcd(n, M)); ker = [(M//g)*(1-n)] + [M//g]*(n-1)
        rng = np.random.default_rng(3); th = 0.37
        R = lambda a, b: o.logW(th, [x+y for x, y in zip(a, ker)], b) - o.logW(th, a, b)
        cross = 0
        for _ in range(6):
            a, b, a2, b2 = (rnd(rng, n, M) for _ in range(4))
            cross = max(cross, abs(wrap(R(a, b) + R(a2, b2) - R(a, b2) - R(a2, b))))
        print(f"(n,M)=({n},{M}) kernel shift: gauge cross-ratio deviation {cross:.2e} (nonzero: not a gauge factor)")
    for n, M in [(2, 2), (3, 2), (3, 3), (4, 2), (4, 3)]:
        o = Ours(n, M, T, 0); o.w0 = PI/n; rng = np.random.default_rng(5); dev = devS = devB = 0
        C = lambda a: a[1:] + [a[0]-1]
        th = [0.3, 0.8, 1.2, 0.5]
        box = lambda a, b, c, d: o.logW(th[0], a, b) + o.logW(th[1], b, c) + o.logW(th[2], c, d) - o.logW(th[3], a, d)
        for _ in range(4):
            a, b, c, d = (rnd(rng, n, M) for _ in range(4))
            dev = max(dev, abs(wrap(o.logW(0.37, C(a), C(b)) - o.logW(0.37, a, b))))
            devS = max(devS, abs(wrap(logS(n, M, T, PI/n, C(a)) - logS(n, M, T, PI/n, a) - 1j*PI*(n-1))))
            base = box(a, b, c, d)
            for slot in range(4):
                sp = [list(a), list(b), list(c), list(d)]; sp[slot][0] += M; sp[slot][-1] -= M
                devB = max(devB, abs(wrap(box(*sp) - base)))
        print(f"(n,M)=({n},{M}) C-symmetry W {dev:.1e} S {devS:.1e}; box lift-invariance {devB:.1e}")
