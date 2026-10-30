from lensnp import *
r = 3; tau1 = 0.1+0.6j; tau2 = 0.23+0.4j
z = 0.37+0.05j
for m in range(-2, 5):
    a = logPhi(z, m, r, tau1, tau2, reduce=False); b = logPhi(z, m, r, tau1, tau2)
    print("m", m, "periodicity ratio", np.exp(a-b))
print("inversion", np.exp(logPhi(z,1,r,tau1,tau2)+logPhi(-z,-1,r,tau1,tau2)))
p = np.exp(1j*PI*tau1); q = np.exp(1j*PI*tau2)
for m in range(0, 4):
    lhs = logPhi(z+PI*tau2, m-1, r, tau1, tau2) - logPhi(z, m, r, tau1, tau2)
    w = z + (r/2 - m)*PI*tau1; P = p*q; Q = p**r
    rhs = E(z+PI*tau2, m-1, r,tau1,tau2) - E(z, m, r,tau1,tau2) - np.log(qpoch(np.exp(2j*w)*P*Q, Q**2)*qpoch(np.exp(-2j*w)/P*Q, Q**2))
    print("shift m", m, np.exp(lhs-rhs))
