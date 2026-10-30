"""Z_n-symmetric special solutions: all spins at phi0_p = (p-(n+1)/2) w0, w0 an n-torsion point of C/(pi Z + pi M T Z)."""
import sys
import numpy as np
from lensnp import PI
from modframe import ModModel

def scan(n, M, T, rap=(0.9, 0.5, 0.2, 0.1)):
    mm = ModModel(n, M, T)
    for k in range(n):
        for l in range(n):
            if k == 0 and l == 0:
                continue
            w0 = (PI*k + PI*M*T*l)/n
            P0 = np.array([(p-(n+1)/2)*w0 for p in range(1, n+1)])
            for name, e in [("eps=+", 1), ("eps=-", -1)]:
                eps = [[e]*n for _ in range(n)]
                try:
                    R0, dev = mm.test(rap, [P0]*4, P0, P0, trials=4, eps1=eps, eps2=eps)
                    print("n=%d M=%d (k,l)=(%d,%d) %s : dev=%.1e  R=%s" % (n, M, k, l, name, dev, np.round(R0, 6)))
                except Exception as ex:
                    print("n=%d M=%d (k,l)=(%d,%d) %s : error %s" % (n, M, k, l, name, ex))

if __name__ == "__main__":
    n, M = int(sys.argv[1]), int(sys.argv[2])
    scan(n, M, T=0.1+0.25j)
