"""Fast numpy implementation of the lens elliptic gamma function, conventions of arXiv:1709.07148.

p = e^{i pi tau1}, q = e^{i pi tau2};
Phi(z,m) = E(z,m) Phi1(z+(r/2-m) pi tau1; pq, p^r) Phi1(z-(r/2-m) pi tau2; pq, q^r)
E(z,m)   = exp( i m(r-m)/(6r) (6z + pi (r-2m)(tau1-tau2-1)) )
Phi1(z;p,q) = prod_{j,k>=0} (1-e^{2iz}p^{2j+1}q^{2k+1})/(1-e^{-2iz}p^{2j+1}q^{2k+1})
"""
import numpy as np

PI = np.pi


def logPhi1_series(z, p, q, nterms=None, tol=1e-17):
    """sum_{n>=1} (e^{-2inz}-e^{2inz})(pq)^n/(n(1-p^{2n})(1-q^{2n})). Needs |e^{+-2iz}pq|<1."""
    z = np.asarray(z, dtype=complex)
    rate = max(abs(np.exp(2j * z) * p * q).max(), abs(np.exp(-2j * z) * p * q).max())
    if rate >= 1:
        raise ValueError("series does not converge, rate=%g" % rate)
    if nterms is None:
        nterms = int(np.log(tol) / np.log(rate)) + 5
    n = np.arange(1, nterms + 1)
    coef = (p * q) ** n / (n * (1 - p ** (2 * n)) * (1 - q ** (2 * n)))
    zz = z[..., None]
    return np.sum((np.exp(-2j * n * zz) - np.exp(2j * n * zz)) * coef, axis=-1)


def logPhi1_prod(z, p, q, tol=1e-18):
    """double product (only when |p|,|q| small)."""
    z = np.asarray(z, dtype=complex)
    J = int(np.log(tol) / np.log(abs(p))) // 2 + 2
    K = int(np.log(tol) / np.log(abs(q))) // 2 + 2
    j = np.arange(J)[:, None]
    k = np.arange(K)[None, :]
    t = (p ** (2 * j + 1) * q ** (2 * k + 1)).ravel()
    zz = z[..., None]
    return np.sum(np.log(1 - np.exp(2j * zz) * t) - np.log(1 - np.exp(-2j * zz) * t), axis=-1)


def E(z, m, r, tau1, tau2):
    return 1j * m * (r - m) / (6.0 * r) * (6 * z + PI * (r - 2 * m) * (tau1 - tau2 - 1))


def logPhi(z, m, r, tau1, tau2, reduce=True):
    if reduce:
        m = int(m) % r
    p = np.exp(1j * PI * tau1)
    q = np.exp(1j * PI * tau2)
    return (E(z, m, r, tau1, tau2)
            + logPhi1_prod(z + (r / 2 - m) * PI * tau1, p * q, p ** r)
            + (logPhi1_prod if abs(q ** r) < 0.8 else logPhi1_series)(z - (r / 2 - m) * PI * tau2, p * q, q ** r))


# normalised theta functions (1709, App. A):  bar-theta_j(x|tau)
def logthbar(j, x, tau, tol=1e-18):
    x = np.asarray(x, dtype=complex)
    Q = np.exp(1j * PI * tau)
    nmax = int(np.log(tol) / np.log(abs(Q))) // 2 + 3
    n = np.arange(1, nmax)
    if j in (1, 2):
        t = Q ** (2 * n)
    else:
        t = Q ** (2 * n - 1)
    sgn = -1 if j in (1, 4) else 1
    xx = x[..., None]
    s = np.sum(np.log(1 + sgn * np.exp(2j * xx) * t) + np.log(1 + sgn * np.exp(-2j * xx) * t), axis=-1)
    if j == 1:
        s = s + np.log(2 * np.sin(x)) + 1j * PI * tau / 4
    elif j == 2:
        s = s + np.log(2 * np.cos(x)) + 1j * PI * tau / 4
    return s


def qpoch(x, Q, tol=1e-18):
    x = np.asarray(x, dtype=complex)
    nmax = int(np.log(tol) / np.log(abs(Q))) + 3
    n = np.arange(nmax)
    return np.prod(1 - x[..., None] * Q ** n, axis=-1)
