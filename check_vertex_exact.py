from lensnp import *
# pair(x,m) = Phi(-i eta + x, m) Phi(-i eta - x, -m)  (vertex factor for one pair i<j)
def logpair(x, m, r, t1, t2):
    eta = -1j*PI*(t1+t2)/2
    return logPhi(-1j*eta + x, m, r, t1, t2) + logPhi(-1j*eta - x, -m, r, t1, t2)
# first check the exact theta-form (1709 bwdef_S, with 2x->x, 2m->m):
# pair = exp(eta r/2 - 2 eta m^2/r) * thbar1(x - pi m t1 | r t1) * thbar1(x + pi m t2| r t2) * const?
r = 3; t1 = 0.1+0.5j; t2 = 0.05+0.4j; eta = -1j*PI*(t1+t2)/2
x = 0.3+0.02j
for m in range(-2, 4):
    lhs = logpair(x, m, r, t1, t2)
    rhs = eta*r/2 - 2*eta*m**2/r + logthbar(1, x - PI*m*t1, r*t1) + logthbar(1, x + PI*m*t2, r*t2)
    print("m", m, "exp(lhs-rhs)=", np.exp(lhs-rhs))
