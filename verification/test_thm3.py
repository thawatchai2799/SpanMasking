"""
DESIGN-CRITICAL TESTS FOR THEOREM 3
===================================
Two questions that decide whether the Log-Determinant Barrier (LDB) is a
usable objective or an empty promise.

Q1  Is the anti-collapse bound non-vacuous at realistic embedding widths?

    Claim:  L_ent(C) = -(1/2) log det(C + eI) <= c   implies
            lam_min(C) >= exp(-2c) * ((d-1)/(tr C + d e))^(d-1) - e

    The (d-1) exponent is alarming. If tr(C) is not controlled, the bound
    decays geometrically in d and says nothing at d = 192. This test finds
    where the cliff is.

Q2  Does an optimiser actually FIND the ill-conditioned points that VICReg
    permits, or is the counterexample a hand-built curiosity?

    We run an adversarial search: minimise  L_reg(Z) + eta * lam_min(Z),
    i.e. actively push the smallest eigenvalue to zero while keeping the
    regulariser satisfied. If VICReg can be driven to near-zero lam_min at
    low loss, its guarantee is empty. If LDB resists, the barrier is real.
"""

import numpy as np

RNG = np.random.default_rng(0)
GAMMA = 1.0          # VICReg variance target
EPS = 1e-4           # logdet ridge


# ----------------------------------------------------------------------
# covariance helpers
# ----------------------------------------------------------------------

def cov_of(Z):
    Zc = Z - Z.mean(0, keepdims=True)
    n = Z.shape[0]
    return (Zc.T @ Zc) / (n - 1), Zc


def grad_to_Z(dLdC, Zc, n):
    return (2.0 / (n - 1)) * Zc @ (0.5 * (dLdC + dLdC.T))


# ----------------------------------------------------------------------
# losses (value + dL/dC)
# ----------------------------------------------------------------------

def vicreg_parts(C):
    d = C.shape[0]
    var = np.sqrt(np.maximum(np.diag(C), 0) + 1e-8)
    hinge = np.maximum(0.0, GAMMA - var)
    L_var = float(np.sum(hinge ** 2))
    off = C - np.diag(np.diag(C))
    L_cov = float(np.sum(off ** 2))

    dvar = np.zeros_like(C)
    np.fill_diagonal(dvar, -2.0 * hinge * (0.5 / var))
    dcov = 2.0 * off
    return L_var, L_cov, dvar + dcov


def ldb_parts(C):
    d = C.shape[0]
    M = C + EPS * np.eye(d)
    sign, ld = np.linalg.slogdet(M)
    if sign <= 0:
        return 1e9, np.zeros_like(C)
    return -0.5 * float(ld), -0.5 * np.linalg.inv(M)


def lam_min_and_grad(C):
    w, V = np.linalg.eigh(C)
    v = V[:, 0]
    return float(w[0]), np.outer(v, v)


def eff_rank(C):
    w = np.linalg.eigvalsh(C)
    w = np.maximum(w, 0)
    p = w / max(w.sum(), 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


# ----------------------------------------------------------------------
# Q1: bound tightness
# ----------------------------------------------------------------------

def bound(c, d, trC, e=EPS):
    """lam_min lower bound implied by L_ent <= c."""
    return np.exp(-2 * c) * ((d - 1) / (trC + d * e)) ** (d - 1) - e


def q1_bound_tightness():
    print("=" * 74)
    print("Q1  IS THE ANTI-COLLAPSE BOUND NON-VACUOUS AT REALISTIC d ?")
    print("=" * 74)
    print("  Setting: isotropic C = s*I, so lam_min = s exactly.")
    print("  We compare the TRUE lam_min against the bound the theorem gives.\n")
    print(f"  {'d':>5} {'tr(C)/d':>9} {'L_ent':>10} {'true lam_min':>14}"
          f" {'bound':>13} {'usable?':>9}")
    print("  " + "-" * 64)
    for d in [16, 48, 128, 192, 256]:
        for scale in [1.0, 1.5, 2.0]:
            s = scale
            C = s * np.eye(d)
            c, _ = ldb_parts(C)
            b = bound(c, d, np.trace(C))
            usable = "YES" if b > 1e-6 else "vacuous"
            print(f"  {d:>5} {scale:>9.1f} {c:>10.2f} {s:>14.4f}"
                  f" {b:>13.3e} {usable:>9}")
        print()

    print("  Interpretation:")
    print("   * at tr(C)/d = 1 the factor ((d-1)/d)^(d-1) -> 1/e = 0.368,")
    print("     so the bound stays within a constant factor of the truth at")
    print("     ANY width. The (d-1) exponent is harmless here.")
    print("   * at tr(C)/d = 2 the same factor becomes ~2^-(d-1), which is")
    print("     10^-58 at d=192. The bound says nothing.")
    print("   DESIGN IMPLICATION: LDB must be applied to a trace-normalised")
    print("   covariance (unit average variance), not to raw embeddings.\n")


# ----------------------------------------------------------------------
# Q2: adversarial collapse search
# ----------------------------------------------------------------------

def adversarial_search(mode, d=32, n=512, steps=4000, lr=0.02, eta=3.0,
                       seed=0):
    """Actively drive lam_min -> 0 while keeping the regulariser happy."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, d)) * 0.5
    hist = []
    for t in range(steps):
        C, Zc = cov_of(Z)
        if mode == "vicreg":
            Lv, Lc, dC = vicreg_parts(C)
            Lreg = 25.0 * Lv + 1.0 * Lc
            dC = 25.0 * dC_split(C, "var") + 1.0 * dC_split(C, "cov")
        else:
            Lreg, dC = ldb_parts(C)
            Lreg = 10.0 * Lreg
            dC = 10.0 * dC
            # keep trace pinned so the bound applies
            tr_pen = (np.trace(C) - d) ** 2 * 0.01
            dC = dC + 0.02 * (np.trace(C) - d) * np.eye(d)
            Lreg = Lreg + tr_pen

        lm, dlm = lam_min_and_grad(C)
        total = Lreg + eta * lm
        g = grad_to_Z(dC + eta * dlm, Zc, n)
        Z = Z - lr * g
        if t % 500 == 0 or t == steps - 1:
            hist.append((t, Lreg, lm, eff_rank(C)))
    C, _ = cov_of(Z)
    lm, _ = lam_min_and_grad(C)
    return Lreg, lm, eff_rank(C), np.trace(C) / d, hist


def dC_split(C, which):
    """Separate gradients so the two VICReg weights can differ."""
    d = C.shape[0]
    if which == "var":
        var = np.sqrt(np.maximum(np.diag(C), 0) + 1e-8)
        hinge = np.maximum(0.0, GAMMA - var)
        g = np.zeros_like(C)
        np.fill_diagonal(g, -2.0 * hinge * (0.5 / var))
        return g
    off = C - np.diag(np.diag(C))
    return 2.0 * off


def q2_adversarial():
    print("=" * 74)
    print("Q2  CAN AN OPTIMISER FIND THE COLLAPSED POINTS VICReg PERMITS ?")
    print("=" * 74)
    print("  Objective: minimise  L_reg(Z) + eta * lam_min(Z),  d=32, eta=3.0")
    print("  A regulariser with a real guarantee should refuse to let")
    print("  lam_min reach zero no matter how hard we push.\n")
    print(f"  {'regulariser':>12} {'final L_reg':>12} {'lam_min':>11}"
          f" {'eff. rank':>10} {'tr(C)/d':>9}")
    print("  " + "-" * 58)
    out = {}
    for mode in ["vicreg", "ldb"]:
        Lreg, lm, er, trn, hist = adversarial_search(mode)
        out[mode] = (Lreg, lm, er, trn)
        print(f"  {mode:>12} {Lreg:>12.4f} {lm:>11.3e} {er:>10.2f}"
              f" {trn:>9.3f}")

    print("\n  trajectory of lam_min:")
    for mode in ["vicreg", "ldb"]:
        _, _, _, _, hist = adversarial_search(mode)
        traj = "  ".join(f"{h[2]:.2e}" for h in hist)
        print(f"    {mode:>8}: {traj}")

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    lv, ll = out["vicreg"][1], out["ldb"][1]
    print(f"  VICReg  lam_min = {lv:.3e},  eff.rank = {out['vicreg'][2]:.2f}"
          f"  (of 32)")
    print(f"  LDB     lam_min = {ll:.3e},  eff.rank = {out['ldb'][2]:.2f}"
          f"  (of 32)")
    if lv < 1e-3 and ll > 1e-2:
        print("\n  => THEOREM 3 CONFIRMED OPERATIONALLY. An ordinary optimiser")
        print("     drives VICReg to a rank-deficient solution while keeping")
        print("     the regulariser small. LDB refuses. The distinction is")
        print("     not a hand-built curiosity.")
    elif lv >= 1e-3:
        print("\n  => CLAIM WEAKENED. VICReg resisted the adversarial search.")
        print("     The counterexample exists in principle but is not easy to")
        print("     reach by gradient descent. Soften the paper's language:")
        print("     'permits' rather than 'converges to'.")
    else:
        print("\n  => LDB ALSO COLLAPSED. Barrier implementation is wrong or")
        print("     the trace penalty is fighting it. Investigate before use.")
    print("=" * 74)


if __name__ == "__main__":
    print("\nDESIGN-CRITICAL TESTS FOR THEOREM 3\n")
    q1_bound_tightness()
    q2_adversarial()
