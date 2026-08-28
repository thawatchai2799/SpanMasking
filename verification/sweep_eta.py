"""
CRITICAL-PRESSURE SWEEP  (corrected Q2)
=======================================
The first adversarial test used a single collapse pressure eta = 3.0 and
VICReg survived. That was a badly designed test, not evidence of robustness:
the analytic counterexample costs VICReg about 4.8 in loss, so a reward of
only 3.0 for collapsing was never enough to make it worthwhile.

The right question is not "does VICReg collapse at one arbitrary pressure"
but "is there ANY finite pressure at which it collapses". Theory says:

  * VICReg is a bounded penalty. Its cost for full collapse is finite, call
    it K. Any eta > K makes collapse profitable. A finite threshold eta*
    must exist.

  * LDB is a barrier. Its cost for collapse is -0.5*log(lam_min) -> +infinity
    while the reward eta*lam_min -> 0. No finite eta can make collapse
    profitable. No threshold should exist.

So we sweep eta and look for the threshold. Finding one for VICReg and none
for LDB is the operational content of Theorem 3.
"""

import numpy as np

GAMMA = 1.0
EPS = 1e-4
W_VAR, W_COV = 25.0, 1.0


def cov_of(Z):
    Zc = Z - Z.mean(0, keepdims=True)
    return (Zc.T @ Zc) / (Z.shape[0] - 1), Zc


def grad_to_Z(dLdC, Zc, n):
    return (2.0 / (n - 1)) * Zc @ (0.5 * (dLdC + dLdC.T))


def vicreg(C):
    var = np.sqrt(np.maximum(np.diag(C), 0) + 1e-8)
    hinge = np.maximum(0.0, GAMMA - var)
    off = C - np.diag(np.diag(C))
    L = W_VAR * float(np.sum(hinge ** 2)) + W_COV * float(np.sum(off ** 2))
    g = np.zeros_like(C)
    np.fill_diagonal(g, W_VAR * (-2.0 * hinge * (0.5 / var)))
    return L, g + W_COV * 2.0 * off


def ldb(C):
    d = C.shape[0]
    M = C + EPS * np.eye(d)
    sign, ld = np.linalg.slogdet(M)
    if sign <= 0:
        return 1e9, np.zeros_like(C)
    return -0.5 * float(ld), -0.5 * np.linalg.inv(M)


def lam_min(C):
    w, V = np.linalg.eigh(C)
    return float(w[0]), np.outer(V[:, 0], V[:, 0])


def eff_rank(C):
    w = np.maximum(np.linalg.eigvalsh(C), 0)
    p = w / max(w.sum(), 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


def run(mode, eta, d=32, n=512, steps=6000, lr=0.02, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, d)) * 0.5
    reg = vicreg if mode == "vicreg" else ldb
    for _ in range(steps):
        C, Zc = cov_of(Z)
        L, dC = reg(C)
        lm, dlm = lam_min(C)
        Z = Z - lr * grad_to_Z(dC + eta * dlm, Zc, n)   # +eta: minimising +eta*lam_min pushes it DOWN
    C, _ = cov_of(Z)
    lm, _ = lam_min(C)
    return reg(C)[0], lm, eff_rank(C)


def main():
    print("\nCRITICAL-PRESSURE SWEEP")
    print("=" * 74)
    print("  d=32, n=512, 6000 steps. Objective: L_reg(Z) + eta*lam_min(Z)")
    print("  (negative sign = the optimiser is REWARDED for shrinking")
    print("   the smallest eigenvalue, i.e. for collapsing a direction)\n")

    etas = [1, 3, 6, 10, 20, 40, 80, 160]
    print(f"  {'eta':>6} | {'VICReg lam_min':>15} {'eff.rank':>9}"
          f" | {'LDB lam_min':>13} {'eff.rank':>9}")
    print("  " + "-" * 62)
    res = {}
    for eta in etas:
        _, lv, ev = run("vicreg", eta)
        _, ll, el = run("ldb", eta)
        res[eta] = (lv, ll)
        flag_v = "  <-- COLLAPSED" if lv < 1e-3 else ""
        print(f"  {eta:>6} | {lv:>15.3e} {ev:>9.2f}"
              f" | {ll:>13.3e} {el:>9.2f}{flag_v}")

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    v_collapsed = [e for e in etas if res[e][0] < 1e-3]
    l_collapsed = [e for e in etas if res[e][1] < 1e-3]
    print(f"  VICReg collapsed at eta = {v_collapsed if v_collapsed else 'never (within sweep)'}")
    print(f"  LDB    collapsed at eta = {l_collapsed if l_collapsed else 'never'}")
    print()
    if v_collapsed and not l_collapsed:
        print(f"  => THEOREM 3 CONFIRMED OPERATIONALLY.")
        print(f"     VICReg has a finite collapse threshold near eta* ="
              f" {min(v_collapsed)}.")
        print(f"     LDB has none across a {max(etas)//min(etas)}x range of")
        print(f"     pressure, as a barrier must.")
        print(f"     Paper wording: VICReg's guarantee is conditional on the")
        print(f"     competing gradient pressure staying below a finite")
        print(f"     threshold that the objective itself does not control.")
    elif not v_collapsed:
        print("  => NO THRESHOLD FOUND for VICReg within this sweep. Either")
        print("     the pressure range is still too small, or gradient")
        print("     descent cannot reach the counterexample region. Report")
        print("     the sublevel-set result as a statement about the loss")
        print("     landscape only, NOT about optimiser behaviour.")
    else:
        print("  => LDB ALSO COLLAPSED. Barrier implementation is wrong.")
    print("=" * 74)


if __name__ == "__main__":
    main()
