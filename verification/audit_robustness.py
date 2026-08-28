"""
ROBUSTNESS AUDIT OF PROPOSITIONS 5 AND 6
========================================
Both headline results rest on a single seed, a single width, and beta = 1.
beta appears in both formulas but has never been varied, so a wrong beta
factor would be invisible. This checks four axes:

  A  beta in {0.25, 0.5, 1, 2, 4}   -- does lam* = beta/(2 eta) - eps hold?
  B  seeds 0..4                     -- is the equilibrium seed-independent?
  C  d in {16, 32, 64}              -- does width matter?
  D  n in {128, 384, 1024}          -- does batch size matter?

A failure on axis A would mean the beta factor in Propositions 5 and 6 is
wrong. Failures on B-D would mean the equilibrium is an artefact of one
configuration rather than a property of the objective.
"""

import numpy as np

EPS_DEFAULT = 1e-3          # larger eps -> faster convergence, same physics


def cov_of(Z):
    Zc = Z - Z.mean(0, keepdims=True)
    return (Zc.T @ Zc) / (Z.shape[0] - 1), Zc


def step(Z, eta, eps, beta):
    C, Zc = cov_of(Z)
    d = C.shape[0]
    M = C + eps * np.eye(d)
    dC = -(beta / 2.0) * np.linalg.inv(M)
    w, V = np.linalg.eigh(C)
    dC = dC + eta * np.outer(V[:, 0], V[:, 0])
    g = (2.0 / (Z.shape[0] - 1)) * Zc @ (0.5 * (dC + dC.T))
    return g, float(w[0])


def equilibrium(eta, eps=EPS_DEFAULT, beta=1.0, d=32, n=384,
                steps=6000, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, d)) * 0.5
    lr = min(0.02, 1.0 / eta)
    for _ in range(steps):
        g, _ = step(Z, eta, eps, beta)
        Z = Z - lr * g
    return step(Z, eta, eps, beta)[1]


def report(name, rows, predict):
    print(f"\n  {name}")
    print(f"    {'setting':>16} {'predicted':>12} {'measured':>12} {'rel.err':>9}")
    print("    " + "-" * 53)
    worst = 0.0
    for label, kw in rows:
        p = predict(**kw)
        m = equilibrium(**kw)
        e = abs(m - p) / p * 100 if p > 0 else float("nan")
        worst = max(worst, e)
        print(f"    {label:>16} {p:>12.6f} {m:>12.6f} {e:>8.3f}%")
    print(f"    worst relative error: {worst:.3f}%")
    return worst


def main():
    print("\nROBUSTNESS AUDIT OF PROPOSITIONS 5 AND 6")
    print("=" * 70)
    print(f"  law under test:  lam* = beta/(2*eta) - eps      (eps = {EPS_DEFAULT:.0e})")

    pred = lambda eta, eps=EPS_DEFAULT, beta=1.0, **k: beta / (2 * eta) - eps

    worsts = {}

    # --- A: beta (never varied before)
    rows = [(f"beta={b}", dict(eta=25.0, beta=b)) for b in (0.25, 0.5, 1.0, 2.0, 4.0)]
    worsts["A beta"] = report("A. beta sweep at eta = 25", rows, pred)

    # --- B: seeds
    rows = [(f"seed={s}", dict(eta=25.0, seed=s)) for s in range(5)]
    worsts["B seed"] = report("B. seed sweep at eta = 25, beta = 1", rows, pred)

    # --- C: width
    rows = [(f"d={d}", dict(eta=25.0, d=d)) for d in (16, 32, 64)]
    worsts["C width"] = report("C. width sweep", rows, pred)

    # --- D: batch size
    rows = [(f"n={n}", dict(eta=25.0, n=n)) for n in (128, 384, 1024)]
    worsts["D batch"] = report("D. batch-size sweep", rows, pred)

    # --- E: Proposition 6 threshold scales with beta
    print("\n  E. Proposition 6:  eta* = beta/(2*eps)   (eps = 1e-3)")
    print(f"    {'beta':>8} {'predicted eta*':>15} {'lam at 0.5x':>13}"
          f" {'lam at 2x':>13} {'consistent':>12}")
    print("    " + "-" * 65)
    okE = True
    for beta in (0.5, 1.0, 2.0):
        star = beta / (2 * EPS_DEFAULT)
        lo = equilibrium(0.5 * star, beta=beta)
        hi = equilibrium(2.0 * star, beta=beta)
        good = lo > 1e-5 and hi < 1e-5
        okE = okE and good
        print(f"    {beta:>8} {star:>15.0f} {lo:>13.3e} {hi:>13.3e}"
              f" {'yes' if good else 'NO':>12}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    for k, v in worsts.items():
        flag = "OK" if v < 2.0 else "FAIL"
        print(f"  [{flag}] {k:<10} worst error {v:.3f}%")
    print(f"  [{'OK' if okE else 'FAIL'}] E beta-scaling of the collapse threshold")
    if max(worsts.values()) < 2.0 and okE:
        print("\n  => Propositions 5 and 6 hold across all four axes, including")
        print("     the beta factor that had never been tested.")
    else:
        print("\n  => At least one axis fails. The formulas as stated in")
        print("     Section 3.5 are not general and must be restricted.")
    print("=" * 70)


if __name__ == "__main__":
    main()
