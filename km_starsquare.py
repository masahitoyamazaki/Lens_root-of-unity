import sys, itertools, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from lensnp import PI
from quano_fujii import wrap
from qf_check import logW, logS, spins_of
T = 0.13 + 0.55j
for n, M in [(2, 3), (3, 2), (2, 4)]:
    w0 = PI/n
    spins = spins_of(n, M)
    lW = lambda t, A, B: logW(n, M, T, w0, t, A, B)
    lS = lambda A: logS(n, M, T, w0, A)
    u, up, v, vp = 0.9, 0.5, 0.2, 0.1
    Wb = lambda t, A, B: lW(PI/2 - t, A, B)
    def star1(I, J, K, L):
        return np.log(sum(np.exp(lS(h) + Wb(u-v, K, h) + Wb(up-vp, J, h) + lW(up-v, h, I) + lW(u-vp, h, L)) for h in spins))
    rng = np.random.default_rng(1); dIL = dJK = d3 = 0
    for _ in range(6):
        I, I2, J, J2, K, K2, L, L2 = [spins[x] for x in rng.integers(0, len(spins), 8)]
        dIL = max(dIL, abs(wrap(star1(I,J,K,L) - star1(I2,J,K,L) - star1(I,J,K,L2) + star1(I2,J,K,L2))))
        dJK = max(dJK, abs(wrap(star1(I,J,K,L) - star1(I,J2,K,L) - star1(I,J,K2,L) + star1(I,J2,K2,L))))
        dIJ = abs(wrap(star1(I,J,K,L) - star1(I2,J,K,L) - star1(I,J2,K,L) + star1(I2,J2,K,L)))
        d3 = max(d3, dIJ)
    print(n, M, "mixed diff (I,L): %.1e  (J,K): %.1e  (I,J) [edge, expected nonzero]: %.1e" % (dIL, dJK, d3))
