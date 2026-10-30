"""(i) n=2: our saddle-point equation (saddlemod) vs. 1709's Q4 three-leg building block
  Psi(theta|phi_i,phi_j) = log th3(theta+(phi_i+phi_j))/th3(theta-(phi_i+phi_j)) + log th2(theta+(phi_i-phi_j))/th2(theta-(phi_i-phi_j)),
nome r tau' = M T, with phi^KY = phi_ours + pi M T/4 (component 1; component 2 = -component 1)."""
import numpy as np
from lensnp import logthbar, PI

def wrap(x): return x - 2j * PI * np.round(x.imag / (2 * PI))

def ours_leg(th, ph, X, MT, sg):
    """contribution of one leg to G_1 - G_2 (n=2), phi_h = (ph,-ph), X = (x,-x)"""
    g = []
    for p, php in enumerate([ph, -ph]):
        s = 0
        for xi in [X, -X]:
            if sg > 0:
                s += logthbar(2, php - xi + th, MT)
            else:
                s -= logthbar(2, xi - php + th, MT)
        g.append(s)
    return g[0] - g[1]

def Psi(th, pi_, pj, MT):
    return (logthbar(3, th + (pi_ + pj), MT) - logthbar(3, th - (pi_ + pj), MT)
            + logthbar(2, th + (pi_ - pj), MT) - logthbar(2, th - (pi_ - pj), MT))

rng = np.random.default_rng(0)
for MT in [0.2 + 0.7j, -0.1 + 0.45j]:
    diffs = []
    for _ in range(8):
        ph, X = rng.normal(size=2) * 0.3 + 1j * rng.normal(size=2) * 0.05
        th = rng.uniform(0.1, 1.2)
        sh = PI * MT / 4
        for sg in [+1, -1]:
            o = ours_leg(th, ph, X, MT, sg)
            # 1709 convention: leg contributes +Psi(theta|phi_h, phi_X) for W-type, and Psi(eta-theta|...) is already in th
            k = Psi(th, ph + sh, X + sh, MT) if sg > 0 else Psi(th, ph + sh, X + sh, MT)
            diffs.append((sg, wrap(o - sg * k) if sg > 0 else wrap(o - k)))
    for sg in [+1, -1]:
        d = np.array([x for s, x in diffs if s == sg])
        print("MT=%s  leg type %s : ours - (+-)Psi_KY  =  %s ... spread %.1e" % (MT, "W" if sg > 0 else "Wbar", np.round(d[0], 6), np.max(abs(wrap(d - d[0])))))
