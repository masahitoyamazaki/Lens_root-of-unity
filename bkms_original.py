"""Star-star relation with the ORIGINAL BKMS weights (CMP 138 (1991) 393, eqs. (0.4)-(0.7)):
  g_pq(m, m+delta_a) = [h+_{a-1}(p)h-_{a-1}(q) - h-_{a-1}(p)h+_{a-1}(q) w^{m_{a-1,a}}] / [h+_a(p)h-_a(q) - h-_a(p)h+_a(q) w^{1+m_{a,a+1}}]
  Wbar_pq(l,m)/Wbar_pq(0,0) = w^{Q(l,m)} g_pq(0, l-m),   Q(l,m) = sum_a m_{a-1,a} (l_a - m_a),   indices a in Z_n.
Identification: our spin a (sum 0) <-> original m with m_a - m_{a+1} = a_a;  h^+- = z^-+;  (p,q) = (l1,l2);
edges W(l1,l2;A,B) = Wbar_{l1 l2}(A,B), except the (u,v') edge = 1/Wbar_{v' u}(B,A)  (Fig. 2 of BKMS)."""
import itertools
import numpy as np
from lensnp import PI
from bkms_literal import setup, star_star

def orig_spin(a):
    """original m (m_0 = 0) with m_al - m_{al+1} = a_al (cyclic; consistent since sum a = 0)."""
    n = len(a); m = [0]
    for al in range(n - 1):
        m.append(m[-1] - a[al])
    return m

def make(P):
    n, M = P.n, P.M
    om = np.exp(2j * PI / M)
    hp = {l: P.zm[l] for l in P.zm}; hm = {l: P.zp[l] for l in P.zp}
    def logg_step(p, q, m, al):
        """log g_pq(m, m + delta_al)"""
        a1 = (al - 1) % n; a2 = (al + 1) % n
        num = hp[p][a1] * hm[q][a1] - hm[p][a1] * hp[q][a1] * om ** (m[a1] - m[al])
        den = hp[p][al] * hm[q][al] - hm[p][al] * hp[q][al] * om ** (1 + m[al] - m[a2])
        return np.log(num) - np.log(den)
    def logg(p, q, D):
        """log g_pq(0, D) along a path of unit steps (negative steps use inverses)."""
        m = [0] * n; val = 0
        for al in range(n):
            for _ in range(abs(D[al])):
                if D[al] > 0:
                    val += logg_step(p, q, m, al); m[al] += 1
                else:
                    m[al] -= 1; val -= logg_step(p, q, m, al)
        return val
    def logW(p, q, l, m):
        Q = sum((m[(al - 1) % n] - m[al]) * (l[al] - m[al]) for al in range(n))
        return 2j * PI / M * Q + logg(p, q, [l[i] - m[i] for i in range(n)])
    return logW

if __name__ == "__main__":
    for (n, M) in [(2, 3), (2, 5), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (5, 2)]:
        out = []
        for seed in range(3):
            P, rng = setup(n, M, seed)
            W = make(P)
            def E(l1, l2, A, B):
                if (l1, l2) == ("u", "vp"):
                    return -W("vp", "u", orig_spin(B), orig_spin(A))
                return W(l1, l2, orig_spin(A), orig_spin(B))
            out.append(star_star(P, rng, E, trials=5))
        print("n=%d M=%d : original BKMS weights, star-star devs %s, R=%s"
              % (n, M, ["%.0e" % o[0] for o in out], ["%.4f%+.4fj" % (o[1].real, o[1].imag) for o in out]))
