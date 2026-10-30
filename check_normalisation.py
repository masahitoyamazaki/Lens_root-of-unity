import sys
from normalisation import *
n, r, N = (int(s) for s in sys.argv[1:4])
tau = 0.05+0.35j; rap = (0.9, 0.5, 0.2, 0.1)
ratio, data = run(n, r, N, tau, rap, trials=1, verbose=False, return_data=True)
print("n,r,N=%d,%d,%d  F1/F2 / R_simplified = %s" % (n, r, N, ratio/predict_simplified(n, r, N, tau, rap, *data)))
