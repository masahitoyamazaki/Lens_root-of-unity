"""Numerical check of the A_{n-1} star-star relation (Kels-Yamazaki 1704.03159, Sec. 6)
rewritten in the conventions of 1709.07148 (Phi(z,m), p=e^{i pi tau1}, q=e^{i pi tau2}).
Spin sigma=(x,m): x in R^n, sum x=0 ; m in Z_r^n, sum m = 0 mod r.
W_alpha(sa,sb) = prod_{i,j} Phi(x_{a,i}-x_{b,j}+i alpha, m_{a,i}-m_{b,j})
Wbar_alpha     = W_{eta-alpha}  (the sqrt(S S) factors of 1704 moved onto the internal vertex)
S(s)           = prod_{i != j} Phi(-i eta + x_i - x_j, m_i - m_j)
eta            = -i pi (tau1+tau2)/2
"""
import itertools, sys
import numpy as np
from lensnp import logPhi, PI

def logW(alpha, xa, ma, xb, mb, r, t1, t2):
    s = 0
    for i in range(len(xa)):
        for j in range(len(xb)):
            s = s + logPhi(xa[i] - xb[j] + 1j*alpha, ma[i] - mb[j], r, t1, t2)
    return s

def logS(x, m, eta, r, t1, t2):
    s = 0
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                s = s + logPhi(-1j*eta + x[i] - x[j], m[i] - m[j], r, t1, t2)
    return s

def run(n, r, t1, t2, u, up, v, vp, M=48, seed=1):
    rng = np.random.default_rng(seed)
    eta = -1j*PI*(t1+t2)/2
    assert abs(eta.imag) < 1e-12; eta = eta.real
    def rspin():
        x = rng.uniform(-1, 1, n); x -= x.mean()
        m = list(rng.integers(0, r, n-1)); m.append((-sum(m)) % r)
        return list(x), m
    si, sj, sk, sl = [rspin() for _ in range(4)]
    # grid for internal spin: x_1..x_{n-1} in [0,pi), x_n=-sum
    g = np.arange(M) * PI / M
    grids = np.meshgrid(*([g]*(n-1)), indexing='ij')
    xh = [G.ravel() for G in grids]
    xh.append(-sum(xh))
    W1 = 0; W2 = 0
    for mh in itertools.product(range(r), repeat=n-1):
        mh = list(mh) + [(-sum(mh)) % r]
        S = logS(xh, mh, eta, r, t1, t2)
        l1 = (S + logW(eta-(u-v), sk[0], sk[1], xh, mh, r, t1, t2)
                + logW(eta-(up-vp), sj[0], sj[1], xh, mh, r, t1, t2)
                + logW(up-v, xh, mh, si[0], si[1], r, t1, t2)
                + logW(u-vp, xh, mh, sl[0], sl[1], r, t1, t2))
        l2 = (S + logW(eta-(u-v), xh, mh, sj[0], sj[1], r, t1, t2)
                + logW(eta-(up-vp), xh, mh, sk[0], sk[1], r, t1, t2)
                + logW(up-v, sl[0], sl[1], xh, mh, r, t1, t2)
                + logW(u-vp, si[0], si[1], xh, mh, r, t1, t2))
        W1 = W1 + np.sum(np.exp(l1)); W2 = W2 + np.sum(np.exp(l2))
    lhs = np.exp(logW(vp-v, sl[0], sl[1], sk[0], sk[1], r, t1, t2) + logW(up-u, sl[0], sl[1], sj[0], sj[1], r, t1, t2)) * W1
    rhs = np.exp(logW(vp-v, sj[0], sj[1], si[0], si[1], r, t1, t2) + logW(up-u, sk[0], sk[1], si[0], si[1], r, t1, t2)) * W2
    return lhs, rhs

if __name__ == "__main__":
    n = int(sys.argv[1]); r = int(sys.argv[2]); M = int(sys.argv[3])
    t1, t2 = 0.5j, 0.4j
    lhs, rhs = run(n, r, t1, t2, u=0.9, up=0.5, v=0.2, vp=0.1, M=M)
    print("n=%d r=%d M=%d  lhs=%s rhs=%s  ratio=%s" % (n, r, M, lhs, rhs, lhs/rhs))
