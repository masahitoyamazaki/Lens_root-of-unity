"""Dictionary between the lens elliptic gamma function of arXiv:1704.03159 (Gamma(z,m;sigma,tau), eqs. (2.2)-(2.5))
and Phi(z,m) of arXiv:1709.07148 (lensnp.logPhi).
Claim:  Phi_1709(pi z, m) = Gamma_1704((sigma+tau)/2 - z, m; sigma, tau)  with sigma = tau1, tau = tau2,
so that  Phi_1704(z, m) := Gamma_1704((sigma+tau)/2 - z, -m) = Phi_1709(pi z, -m)."""
import numpy as np
from lensnp import logPhi, PI


def R2(z, m, s, t, r):
    return ((s + t - 2 * z) * (2 * z**2 - 2 * z * (s + t) + s * t * (r**2 + 6 * (m - r) * m) + 1) / (24 * r * s * t)
            - (s - t) * (2 * m - r) * (m - r) * m / (12 * r))


def logGamma1704(z, m, s, t, r, J=60, K=60):
    m = m % r
    p, q = np.exp(2j * PI * s), np.exp(2j * PI * t)
    phie = 2j * PI * (R2(z, 0, s, t, r) + R2(0, m, 0.5, -0.5, r) - R2(z, m, s, t, r))
    val = 0
    for j in range(J):
        for k in range(K):
            val += (np.log(1 - np.exp(-2j * PI * z) * p**(-m) * (p * q)**(j + 1) * p**(r * (k + 1)))
                    - np.log(1 - np.exp(2j * PI * z) * p**m * (p * q)**j * p**(r * k))
                    + np.log(1 - np.exp(-2j * PI * z) * q**(-r + m) * (p * q)**(j + 1) * q**(r * (k + 1)))
                    - np.log(1 - np.exp(2j * PI * z) * q**(r - m) * (p * q)**j * q**(r * k)))
    return phie + val


if __name__ == "__main__":
    t1, t2 = 0.11 + 0.62j, -0.07 + 0.55j
    for r in [1, 2, 3]:
        devs = []
        for m in range(r):
            for z in [0.13 + 0.02j, -0.21 + 0.01j, 0.3 - 0.03j]:
                a = logPhi(PI * z, m, r, t1, t2)
                b = logGamma1704((t1 + t2) / 2 - z, m, t1, t2, r)
                d = a - b
                devs.append(abs(d - 2j * PI * np.round(d.imag / (2 * PI))))
        print("r=%d: max |log Phi_1709(pi z,m) - log Gamma_1704((sigma+tau)/2 - z, m)| = %.1e" % (r, max(devs)))
