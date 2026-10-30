"""Trigonometric limit T -> i oo of the elliptic Z_M^{n-1} KM-type model (Prop. 5.4).
log thbar1(x + c pi T | T) ->  c>0: log(i)  + i pi T/4 - i pi c T - i x
                               c<0: log(-i) + i pi T/4 + i pi c T + i x
                               c=0: log(2 sin x) + i pi T/4
Here T0 is kept as a finite symbol for the divergent constants (they must cancel)."""
import itertools, sys
import numpy as np
from lensnp import PI

class Trig:
    def __init__(self, n, M, nu=0, T0=0.3+1.7j, mode="limit"):
        self.n, self.M, self.nu, self.T0, self.mode = n, M, nu, T0, mode
    def lt1(self, x, c):
        T0 = self.T0
        if abs(c) < 1e-12:
            return np.log(complex(2*np.sin(x))) + 1j*PI*T0/4
        if c > 0:
            return np.log(1j) + 1j*PI*T0/4 - 1j*PI*c*T0 - 1j*x
        return np.log(-1j) + 1j*PI*T0/4 + 1j*PI*c*T0 + 1j*x
    def logW(self, th, A, B):
        n, M, nu = self.n, self.M, self.nu
        s = 0
        for i in range(n):
            for j in range(n):
                d = A[i]-B[j]; c = (i-j)/n
                x0 = (th - (i-j)*PI*nu/n)/M
                if self.mode == "reduced":
                    # conjectured simplified form
                    continue
                if d >= 0:
                    s -= sum(self.lt1(x0+(k+0.5)*PI/M, c) for k in range(d))
                else:
                    s += sum(self.lt1(x0+(k+0.5)*PI/M, c) for k in range(d, 0))
        if self.mode == "reduced":
            for i in range(n):
                d = A[i]-B[i]
                if d >= 0:
                    s -= sum(np.log(complex(np.sin((th+(k+0.5)*PI)/M))) for k in range(d))
                else:
                    s += sum(np.log(complex(np.sin((th+(k+0.5)*PI)/M))) for k in range(d, 0))
            s += -1j*PI/M*sum(A[i]*B[j]-A[j]*B[i] for i in range(n) for j in range(i))
            s += 2j*th/M*sum(i*(A[i]+B[i]) for i in range(n))   # theta-dependent vertex-local part (kept)
        return s
    def logS(self, A):
        n, M, nu = self.n, self.M, self.nu
        if self.mode == "reduced":
            return -2j*PI/M*sum(i*A[i] for i in range(n))
        return sum(self.lt1((PI*(A[i]-A[j]) + (j-i)*PI*nu/n)/M, -(j-i)/n) for i in range(n) for j in range(i+1, n))
    def test(self, rap=(0.9, 0.5, 0.2, 0.1), trials=5, seed=0):
        u, up, v, vp = rap; hf = PI/2; n, M = self.n, self.M
        rng = np.random.default_rng(seed); rats = []
        for t in range(trials):
            A = []
            for _ in range(4):
                a = list(rng.integers(0, M, n-1)) if t else [0]*(n-1); a.append(-sum(a)); A.append(a)
            AI, AJ, AK, AL = A; L = R = 0
            for ah in itertools.product(range(M), repeat=n-1):
                ah = list(ah)+[-sum(ah)]
                L += np.exp(self.logS(ah)+self.logW(hf-(u-v), AK, ah)+self.logW(hf-(up-vp), AJ, ah)+self.logW(up-v, ah, AI)+self.logW(u-vp, ah, AL))
                R += np.exp(self.logS(ah)+self.logW(hf-(u-v), ah, AJ)+self.logW(hf-(up-vp), ah, AK)+self.logW(up-v, AL, ah)+self.logW(u-vp, AI, ah))
            L *= np.exp(self.logW(vp-v, AL, AK)+self.logW(up-u, AL, AJ))
            R *= np.exp(self.logW(vp-v, AJ, AI)+self.logW(up-u, AK, AI))
            rats.append(L/R)
        rats = np.array(rats)
        return rats[0], np.max(abs(rats/rats[0]-1))

if __name__ == "__main__":
    for (n, M) in [(2, 3), (2, 5), (3, 2), (3, 3), (3, 4), (4, 2)]:
        out = []
        for nu in [0, 1]:
            for T0 in [0.3+1.7j, -0.2+0.4j]:
                out.append("%.0e" % Trig(n, M, nu, T0).test()[1])
        r0, dred = Trig(n, M, mode="reduced").test()
        print("n=%d M=%d  limit-form devs (nu,T0)=%s   reduced form dev=%.1e R=%s" % (n, M, out, dred, np.round(r0, 8)))
