# Numerical checks for "Root-of-unity limit of the A_{n-1} lens elliptic gamma function solution of the star-star relation," arXiv:2610.41438 [math-ph] by Masahito Yamazaki

Scripts used for the numerical checks summarized in Appendix F (Table 2) of the paper.

## Requirements and usage

Python 3 with `numpy`, `scipy` and `sympy`. Run each script from this directory, e.g.

    python3 check_Phi_shift.py
    python3 discrete_ssr.py 3 2 2      # n r N

Scripts that take arguments print their usage in the docstring at the top of the file.

## Conventions

Those of the paper, which follow arXiv:1709.07148 (Kels–Yamazaki):
`p = e^{i pi tau1}`, `q = e^{i pi tau2}`, lens elliptic gamma function `Phi(z,m)`.
Root-of-unity limit: `tau2 = 1/(rN) + i hbar/(pi r)`, `lambda = pi (1 + rN tau)/(rN)`, `tauhat = N tau + 1/r`;
modular frame: `M = rN`, `T = r tau/(1 + rN tau)`.
Equation names below are the LaTeX labels of the paper.

## Core module

| file | content |
|---|---|
| `lensnp.py` | `Phi(z,m)`, theta functions `thbar_j`, basic utilities (numpy); imported by almost all scripts |

## Section 2: the model

| script | checks |
|---|---|
| `check_Phi_shift.py` | `Phiprop`, difference equations `shift`, `shift1` |
| `check_vertex_exact.py` | exact vertex factor `pairexact` |
| `dict_1704.py` | dictionary `PhiGamma` between Gamma of arXiv:1704.03159 and Phi |
| `ssr.py n r M` | star-star relation `ssr` by numerical integration |

## Section 3: root-of-unity limit

| script | checks |
|---|---|
| `check_leading_term.py` | leading term `lead` via `keyobs` |
| `omega.py` | cyclic elliptic gamma function `omegalim` (recursive form); also imported as a module |
| `closed_form.py` | closed form `omegadef`, `Pdef` (sympy) |
| `ky_subleading.py` | subleading term `KYsub` for all integers m |
| `psi.py` | a-independent term `psidef`, `psitorsion` |
| `check_vertex_limit.py` | vertex factor `pairasym`, `pairlim` |
| `modular.py` | modular form `omegamod` |

## Section 4: quasi-classical expansion

| script | checks |
|---|---|
| `classical_ssr.py` | classical star-star relation `cssr` |
| `q4check.py` | n=2 three-leg form `q4leg` |
| `discrete_ssr.py n r N` | discrete star-star relation `dssr` (spin independence) |
| `normalisation.py`, `check_normalisation.py n r N` | normalization `Kdef` |

## Section 5: discrete spin solutions

| script | checks |
|---|---|
| `compare1709.py` | n=2 weights vs arXiv:1709.07148 (Sec. 5.1) |
| `modframe.py` | discrete model in the modular frame `dWmod` |
| `r_independence.py n r N` | equivalence `tau1def` of the (r,N,tau) and (1,rN,tilde tau) models |

## Section 6: elliptic sl(n) Kashiwara–Miwa type models

| script | checks |
|---|---|
| `km_real.py` | real Z_n-symmetric configurations `xi0` as saddle points; `dssr` there |
| `symmetric.py n M`, `km_n.py` | weights `KMnW`, `KMnS` and `KMnssr` at torsion points |
| `quano_fujii.py` | Quano–Fujii weights (recursion of their Thm. 4.4) |
| `qf_check.py` | identification `QFid`; `KMnssr` at all primitive torsion points; modular images `QFcond` |
| `qf_ybe.py`, `qf_ybe_fast.py` | Quano–Fujii Yang–Baxter equation `QFYBE` and its reduction to `SSRQF` |
| `km_gcd.py` | gcd(n,M)>1: kernel `kerQF`, cyclic shift `Cshift`, lift invariance of the box weights |
| `km_orbits.py` | orbits of the primitive torsion points (Proposition 1) by enumeration |
| `km_modular.py` | modular covariance (c) of `KMnssr`, numerically |
| `km_modular_symbolic.py`, `km_modular_general_n.py` | symbolic computation for (c) (Appendix E), n<=4 and general n |
| `km_reality.py` | reality property `KMconj` |
| `km_starsquare.py` | the star sum does not factorize into edge weights |

## Section 7: trigonometric limit

| script | checks |
|---|---|
| `trig.py`, `trig2.py` | `trigW`, and the necessity of the chiral phase |
| `trig_bkms2.py`, `trig_bkms3.py` | comparison with the BKMS phase, `trigBKMS` |
| `trig_general.py` | `triggenW`, `triggensaddle`; non-Z-invariant external spins |
| `uniq_symbolic.py`, `uniq_check.py` | saddle point equations as the cross-ratio condition `crcond` |
| `bkms_curve4.py`, `bkms_check.py`, `continuation.py` | rapidities on the curve `bkmscurve`; saddle point tracking (utilities) |
| `bkms_dashed.py`, `bkms_dashed2.py`, `bkms_branch.py` | Z-invariant configurations `zinvspins`; `dssr` on the curve |
| `bkms_ident.py`, `gauge_quad.py`, `gauge_exact.py` | edge-level identification with the BKMS form; gauge factors |
| `bkms_pure.py`, `bkms_final.py` | `dssr` with the weights `BKMSform` |
| `bkms_original.py` | `dssr` with the original BKMS weights (CMP 138 (1991) 393, (0.4)–(0.7)) |
| `bkms_literal.py`, `bkms_variants.py`, `bkms_ident_scan.py` | compressed BKMS form of arXiv:2502.17545 (footnote in Sec. 7.2) |
