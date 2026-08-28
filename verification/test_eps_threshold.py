"""
DOES epsilon SET THE COLLAPSE THRESHOLD OF LDB?
===============================================
The ridge term eps in L_ent = -(beta/2) log det(C + eps*I) was introduced for
numerical stability, but it also destroys the barrier: det(C + eps*I) >= eps^d
is bounded away from zero, so L_ent is BOUNDED and singular covariances do lie
in sufficiently high sublevel sets. LDB with eps > 0 is therefore a penalty,
not a barrier -- exactly the charge Proposition 3 levels at VICReg.

The corrected stationarity condition for -(beta/2)log(lam + eps) + eta*lam is

    lam* = beta/(2*eta) - eps

which is non-positive once eta >= beta/(2*eps). So the theory predicts a sharp
collapse threshold

    eta* = beta / (2*eps)

and, crucially, that eta* moves inversely with eps. If measured thresholds
track that law across three decades of eps, the corrected account is right and
eps becomes a designed safety margin rather than an implementation detail.
"""

import numpy as np

BETA = 1.0
GAMMA = 1.0


def cov_of(Z):
    Zc = Z - Z.mean(0, keepdims=True)
    return (Zc.T @ Zc) / (Z.shape[0] - 1), Zc


def grad_to_Z(dLdC, Zc, n):
    return (2.0 / (n - 1)) * Zc @ (0.5 * (dLdC + dLdC.T))


def ldb(C, eps):
    d = C.shape[0]
    M = C + eps * np.eye(d)
    sign, ld = np.linalg.slogdet(M)
    if sign <= 0:
        return 1e9, np.zeros_like(C)
    return -(BETA / 2.0) * float(ld), -(BETA / 2.0) * np.linalg.inv(M)


def vicreg(C):
    var = np.sqrt(np.maximum(np.diag(C), 0) + 1e-8)
    hinge = np.maximum(0.0, GAMMA - var)
    off = C - np.diag(np.diag(C))
    L = 25.0 * float(np.sum(hinge ** 2)) + float(np.sum(off ** 2))
    g = np.zeros_like(C)
    np.fill_diagonal(g, 25.0 * (-2.0 * hinge * (0.5 / var)))
    return L, g + 2.0 * off


def lam_min(C):
    w, V = np.linalg.eigh(C)
    return float(w[0]), np.outer(V[:, 0], V[:, 0])


def run(eta, eps, mode="ldb", d=24, n=384, steps=9000, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, d)) * 0.5
    lr = min(0.02, 2.0 / eta)          # keep the eta term stable
    for _ in range(steps):
        C, Zc = cov_of(Z)
        L, dC = ldb(C, eps) if mode == "ldb" else vicreg(C)
        lm, dlm = lam_min(C)
        Z = Z - lr * grad_to_Z(dC + eta * dlm, Zc, n)
    C, _ = cov_of(Z)
    return lam_min(C)[0]


def main():
    print("\nDOES eps SET LDB's COLLAPSE THRESHOLD?")
    print("=" * 72)
    print("  prediction:  lam* = beta/(2*eta) - eps,  collapse at "
          "eta* = beta/(2*eps)\n")

    for eps in [1e-2, 1e-3, 1e-4]:
        pred_star = BETA / (2 * eps)
        etas = [pred_star * f for f in (0.2, 0.5, 1.0, 2.0, 5.0)]
        print(f"  eps = {eps:.0e}   predicted eta* = {pred_star:.0f}")
        print(f"    {'eta':>9} {'eta/eta*':>9} {'predicted lam*':>15}"
              f" {'measured':>12} {'status':>11}")
        print("    " + "-" * 60)
        for eta in etas:
            lm = run(eta, eps)
            pred = max(BETA / (2 * eta) - eps, 0.0)
            status = "collapsed" if lm < 1e-6 else "alive"
            ps = f"{pred:.3e}" if pred > 0 else "0 (collapse)"
            print(f"    {eta:>9.0f} {eta/pred_star:>9.1f} {ps:>15}"
                  f" {lm:>12.3e} {status:>11}")
        print()

    print("=" * 72)
    print("CONTROL: VICReg at the same pressures (should collapse far earlier)")
    print("=" * 72)
    print(f"    {'eta':>9} {'VICReg lam_min':>16}")
    for eta in [5, 10, 20, 50]:
        print(f"    {eta:>9} {run(eta, 0.0, mode='vicreg'):>16.3e}")
    print("\n  If LDB's threshold tracks beta/(2 eps) while VICReg's sits near")
    print("  10 regardless, then eps is a designed safety margin and the")
    print("  separation between the two objectives is quantitative and")
    print("  tunable -- not the categorical barrier/penalty split originally")
    print("  claimed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
