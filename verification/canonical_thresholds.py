"""
CANONICAL THRESHOLD SWEEP
=========================
The draft quoted VICReg's collapse threshold as ~10, as [10,40], and as 20,
and LDB's as {50,500,5000} and as {100,1000,10000}. Those came from different
widths, batch sizes, step budgets, grids and collapse criteria, mixed together
in the text. That is indefensible, so every threshold in the paper is
recomputed here under ONE configuration and ONE criterion.

  configuration : d = 24, n = 384, 12000 full-batch steps, seed 0
  pressure      : eta * lam_min  (linear, single direction)
  criterion     : collapsed iff lam_min < 1e-6 at the end of training
  search        : bisection on log(eta), 9 iterations, reported as the
                  bracketing interval [last alive, first collapsed]

Reporting an interval rather than a point is deliberate: the transition is
sharp but its location depends on the step budget, and a single number would
imply a precision the measurement does not have.
"""

import numpy as np

BETA, D, N, STEPS, SEED = 1.0, 24, 384, 12000, 0
CRIT = 1e-6


def run(eta, mode, eps=1e-3, mu=25.0, nu=1.0, gam=1.0):
    rng = np.random.default_rng(SEED)
    Z = rng.standard_normal((N, D)) * 0.5
    lr = min(0.02, 1.0 / max(eta, 1.0))
    I = np.eye(D)
    for _ in range(STEPS):
        Zc = Z - Z.mean(0, keepdims=True)
        C = (Zc.T @ Zc) / (N - 1)
        if mode == "vicreg":
            var = np.sqrt(np.maximum(np.diag(C), 0) + 1e-8)
            hinge = np.maximum(0.0, gam - var)
            dC = np.zeros_like(C)
            np.fill_diagonal(dC, mu * (-2.0 * hinge * (0.5 / var)))
            dC = dC + nu * 2.0 * (C - np.diag(np.diag(C)))
        else:
            dC = -(BETA / 2.0) * np.linalg.inv(C + eps * I)
        w, V = np.linalg.eigh(C)
        dC = dC + eta * np.outer(V[:, 0], V[:, 0])
        Z = Z - lr * (2.0 / (N - 1)) * Zc @ (0.5 * (dC + dC.T))
    Zc = Z - Z.mean(0, keepdims=True)
    return float(np.linalg.eigvalsh((Zc.T @ Zc) / (N - 1))[0])


def bracket(mode, lo=2.0, hi=200000.0, iters=9, **kw):
    if run(lo, mode, **kw) < CRIT:
        return None, lo, run(lo, mode, **kw)
    if run(hi, mode, **kw) >= CRIT:
        return hi, None, run(hi, mode, **kw)
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if run(mid, mode, **kw) < CRIT:
            hi = mid
        else:
            lo = mid
    return lo, hi, run(lo, mode, **kw)


def main():
    print("\nCANONICAL THRESHOLD SWEEP")
    print("=" * 76)
    print(f"  d={D}, n={N}, {STEPS} steps, criterion lam_min < {CRIT:.0e}\n")
    print(f"  {'objective':>9} {'setting':>13} {'predicted':>11}"
          f" {'last alive':>11} {'first dead':>11} {'lam @ alive':>12}")
    print("  " + "-" * 72)

    rows = []
    for eps in (1e-2, 1e-3, 1e-4):
        lo, hi, lam = bracket("ldb", eps=eps)
        pred = BETA / (2 * eps)
        rows.append(("LDB", f"eps={eps:.0e}", pred, lo, hi))
        print(f"  {'LDB':>9} {f'eps={eps:.0e}':>13} {pred:>11.0f}"
              f" {lo:>11.0f} {hi:>11.0f} {lam:>12.2e}")

    for nu in (1.0, 16.0, 256.0, 4096.0):
        lo, hi, lam = bracket("vicreg", nu=nu)
        rows.append(("VICReg", f"nu={nu:g}", None, lo, hi))
        print(f"  {'VICReg':>9} {f'nu={nu:g}':>13} {'--':>11}"
              f" {lo:>11.0f} {hi:>11.0f} {lam:>12.2e}")

    lo, hi, lam = bracket("vicreg", mu=625.0)
    print(f"  {'VICReg':>9} {'mu=625':>13} {'--':>11}"
          f" {lo:>11.0f} {hi:>11.0f} {lam:>12.2e}")

    print("\n" + "=" * 76)
    print("SCALING")
    print("=" * 76)
    ldb = [(r[2], np.sqrt(r[3] * r[4])) for r in rows if r[0] == "LDB"]
    vic = [(float(r[1].split("=")[1]), np.sqrt(r[3] * r[4]))
           for r in rows if r[0] == "VICReg"]
    e_ldb = np.polyfit(np.log([p for p, _ in ldb]),
                       np.log([m for _, m in ldb]), 1)[0]
    e_vic = np.polyfit(np.log([p for p, _ in vic]),
                       np.log([m for _, m in vic]), 1)[0]
    print(f"  LDB   : measured threshold vs beta/(2 eps)  exponent = {e_ldb:+.3f}"
          f"   (theory: +1.000)")
    print(f"  VICReg: measured threshold vs nu            exponent = {e_vic:+.3f}"
          f"   (no theory)")
    ratio = [m for _, m in ldb][-1] / [m for _, m in vic][0]
    print(f"\n  LDB at eps=1e-4 vs VICReg at nu=1: {ratio:.0f}x")
    print("  nu required to match eps=1e-4, extrapolating the fitted exponent:"
          f" {1.0 * ([m for _,m in ldb][-1] / [m for _,m in vic][0]) ** (1/e_vic):.3g}")
    print("=" * 76)
    print("\n  Every threshold quoted in Section 3 must come from this table.")


if __name__ == "__main__":
    main()
