"""n=2: compare our discrete weights (dW),(dS) with arXiv:1709.07148 eqs (4.4)-(4.9).
1709 (after dropping primes): tau' = rN tau/(1+rN tau), phi_i = (zeta_i + r pi tau/4) rN/(1+rN tau),
theta = i rN alpha/(1+rN tau) [= our theta], a_i = r n_i - m_i [= our a_i]."""
import numpy as np
from lensnp import logthbar, PI
from discrete_ssr import Model
from closed_form import wrap

def logw1709(j1, j2, th, phi, a, r, N, tp):
    s = a/(r*N)*(logthbar(j2, phi+th, r*tp) - logthbar(j2, phi-th, r*tp))
    if a >= 1:
        for k in range(1, a+1):
            s += logthbar(j1, (phi-th)/(r*N)+PI*(2*k-1)/(2*r*N), tp/N) - logthbar(j1, (phi+th)/(r*N)+PI*(2*k-1)/(2*r*N), tp/N)
    else:
        for k in range(1, -a+1):
            s += logthbar(j1, (phi+th)/(r*N)-PI*(2*k-1)/(2*r*N), tp/N) - logthbar(j1, (phi-th)/(r*N)-PI*(2*k-1)/(2*r*N), tp/N)
    return s

def logW1709(th, phii, ai, phij, aj, r, N, tp):
    return logw1709(4, 3, th, phii+phij, ai+aj, r, N, tp) + logw1709(1, 2, th, phii-phij, ai-aj, r, N, tp)

def logS1709(phi, a, r, N, tp):
    return logthbar(4, 2*PI*a/(r*N) + 2*phi/(r*N), tp/N)

for (r, N) in [(2, 3), (3, 2), (1, 5), (3, 1)]:
    tau = 0.05+0.35j
    M = Model(2, r, N, tau); lam = M.lam; tp = r*N*tau/(1+r*N*tau)
    phi = lambda z: (z + r*PI*tau/4)*PI/lam
    rng = np.random.default_rng(1)
    errW = 0; Sd = []
    for t in range(30):
        zA, zB = rng.uniform(0.02, 0.2, 2)*PI/N/2   # keep 0 < Re(2N zeta) < pi
        aA, aB = rng.integers(0, r*N, 2)
        th = rng.uniform(-0.5, 1.3)
        ours = M.logW(th, [zA, -zA], [aA, -aA], [zB, -zB], [aB, -aB])
        theirs = logW1709(th, phi(zA), aA, phi(zB), aB, r, N, tp)
        errW = max(errW, abs(wrap(ours-theirs)))
        # vertex: ratio should be a-independent
        Sd.append(wrap(M.logS([zA, -zA], [aA, -aA]) - logS1709(phi(zA), aA, r, N, tp)
                       - (M.logS([zA, -zA], [0, 0]) - logS1709(phi(zA), 0, r, N, tp))))
    print("r,N=%d,%d  max|log W_ours - log W_1709| = %.1e   max vertex deviation = %.1e"
          % (r, N, errW, max(abs(np.array(Sd)))))
