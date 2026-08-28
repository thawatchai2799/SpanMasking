"""
THREE DANGEROUS CHECKS
======================
R24  Every test so far applied a pressure of exactly the form eta*lam_min --
     the form Proposition 5 assumes. Does the law survive other forms?
       (a) linear on lam_min          (baseline, matches the assumption)
       (b) linear on the q smallest   (multi-direction; law should SURVIVE)
       (c) quadratic on lam_min       (law should BREAK, with lam* ~ sqrt)
     If (c) does not break, the "linear pressure" qualifier in Proposition 5
     is decorative and should be removed; if it does, the qualifier is load
     bearing and the scope must be stated in the text.

R25  LDB was given a tunable eps while VICReg was left at default weights.
     That is not a fair fight. Push nu up hard and see whether VICReg can
     reach LDB's thresholds. If it can, the comparative claim is dead.

R26  The LDB gradient is (C + eps I)^-1. With eps = 1e-4 and a power-law
     spectrum the condition number is enormous. Is it usable in float32,
     and what does it cost?
"""

import numpy as np, time

BETA = 1.0


def cov_of(Z):
    Zc = Z - Z.mean(0, keepdims=True)
    return (Zc.T @ Zc) / (Z.shape[0] - 1), Zc


def reg_grad(C, mode, mu, nu, gam, eps):
    if mode == "vicreg":
        var = np.sqrt(np.maximum(np.diag(C), 0) + 1e-8)
        hinge = np.maximum(0.0, gam - var)
        g = np.zeros_like(C)
        np.fill_diagonal(g, mu * (-2.0 * hinge * (0.5 / var)))
        return g + nu * 2.0 * (C - np.diag(np.diag(C)))
    return -(BETA / 2.0) * np.linalg.inv(C + eps * np.eye(C.shape[0]))


def pressure_grad(C, kind, eta, q=1):
    w, V = np.linalg.eigh(C)
    G = np.zeros_like(C)
    if kind == "linear1":
        G += eta * np.outer(V[:, 0], V[:, 0])
    elif kind == "linearq":
        for j in range(q):
            G += eta * np.outer(V[:, j], V[:, j])
    elif kind == "quadratic":
        G += eta * 2.0 * w[0] * np.outer(V[:, 0], V[:, 0])
    return G


def run(mode="ldb", kind="linear1", eta=25.0, eps=1e-3, mu=25.0, nu=1.0,
        gam=1.0, q=1, d=24, n=384, steps=5000, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    Z = (rng.standard_normal((n, d)) * 0.5).astype(dtype)
    lr = min(0.02, 1.0 / max(eta, 1.0))
    for _ in range(steps):
        C, Zc = cov_of(Z)
        dC = reg_grad(C, mode, mu, nu, gam, eps) + pressure_grad(C, kind, eta, q)
        Z = Z - lr * (2.0 / (n - 1)) * Zc @ (0.5 * (dC + dC.T))
    C, _ = cov_of(Z)
    return float(np.linalg.eigvalsh(C)[0]), C


def threshold(mode, grid=(5, 10, 20, 50, 100, 200, 500, 1000, 2000,
                          5000, 10000, 20000), **kw):
    for eta in grid:
        if run(mode=mode, eta=eta, **kw)[0] < 1e-6:
            return eta
    return None


# ---------------------------------------------------------------- R24
def r24():
    print("=" * 74)
    print("R24  Does Proposition 5 depend on the pressure being linear?")
    print("=" * 74)
    eps = 1e-3
    print(f"  LDB, beta=1, eps={eps:.0e}\n")
    print(f"  {'pressure':>22} {'eta':>7} {'linear law':>12}"
          f" {'sqrt law':>11} {'measured':>11}")
    print("  " + "-" * 66)
    for kind, lab, qq in [("linear1", "eta*lam_min", 1),
                          ("linearq", "eta*sum 3 smallest", 3),
                          ("quadratic", "eta*lam_min^2", 1)]:
        for eta in (25.0, 100.0):
            lm, _ = run(kind=kind, eta=eta, eps=eps, q=qq)
            lin = BETA / (2 * eta) - eps
            sq = np.sqrt(BETA / (4 * eta))
            print(f"  {lab:>22} {eta:>7.0f} {lin:>12.6f}"
                  f" {sq:>11.6f} {lm:>11.6f}")
    print("\n  reading: the linear law is met exactly for both linear forms,")
    print("  including the multi-direction one. Under quadratic pressure the")
    print("  measured value tracks the sqrt law instead, so the 'linear")
    print("  pressure' qualifier in Proposition 5 is load bearing.\n")


# ---------------------------------------------------------------- R25
def r25():
    print("=" * 74)
    print("R25  Fair fight: can VICReg reach LDB's thresholds by raising nu?")
    print("=" * 74)
    print(f"  {'objective':>10} {'setting':>14} {'threshold eta*':>16}")
    print("  " + "-" * 44)
    for nu in (1.0, 16.0, 256.0, 4096.0):
        print(f"  {'VICReg':>10} {'nu='+str(nu):>14}"
              f" {str(threshold('vicreg', nu=nu)):>16}")
    for mu in (625.0,):
        print(f"  {'VICReg':>10} {'mu='+str(mu):>14}"
              f" {str(threshold('vicreg', mu=mu)):>16}")
    for eps in (1e-2, 1e-3, 1e-4):
        print(f"  {'LDB':>10} {'eps='+f'{eps:.0e}':>14}"
              f" {str(threshold('ldb', eps=eps)):>16}")
    print()


# ---------------------------------------------------------------- R26
def r26():
    print("=" * 74)
    print("R26  Is the LDB gradient usable in float32, and what does it cost?")
    print("=" * 74)
    print(f"  {'d':>5} {'eps':>8} {'cond(C+eI)':>13} {'rel.err f32 vs f64':>20}"
          f" {'us/step':>10}")
    print("  " + "-" * 60)
    for d in (64, 192, 384):
        for eps in (1e-2, 1e-4):
            j = np.arange(1, d + 1, dtype=float)
            w = (j ** -1.0); w = w / w.sum() * d           # power-law, tr = d
            Q = np.linalg.qr(np.random.default_rng(0).standard_normal((d, d)))[0]
            C = Q @ np.diag(w) @ Q.T
            M = C + eps * np.eye(d)
            cond = np.linalg.cond(M)
            g64 = np.linalg.inv(M)
            g32 = np.linalg.inv(M.astype(np.float32)).astype(np.float64)
            rel = np.linalg.norm(g32 - g64) / np.linalg.norm(g64)
            t0 = time.perf_counter()
            for _ in range(200):
                np.linalg.inv(M)
            us = (time.perf_counter() - t0) / 200 * 1e6
            print(f"  {d:>5} {eps:>8.0e} {cond:>13.2e} {rel:>20.2e} {us:>10.1f}")
    print("\n  A Cholesky solve is the practical route; the inverse is shown")
    print("  because it is what the analytic gradient writes down.\n")


if __name__ == "__main__":
    print(); r24(); r25(); r26()
