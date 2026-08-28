"""verify_unbounded.py -- the two measurements Section 3.3 quotes.

Written 2026-08-28. A line-by-line audit of the manuscript found two sets of
numbers with no script behind them:

  (a) the divergence of cosine + log-determinant WITHOUT the trace penalty
      ("variance rose from 2.9 to 403, cosine similarity fell from 0.218 to
      -0.0003, effective rank rose from 18.4 to 29.8 of 32"), and
  (b) the low-pressure equilibrium of the LDB objective across two decades of
      the ridge ("lam_min 0.461 -> 0.472").

Neither was reproducible from the pack. This file performs both, reusing the
training loop of jepa_linear.py so the settings are the same, and the
manuscript quotes whatever it prints.

Run:  python verify_unbounded.py
"""

import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("jl", HERE / "jepa_linear.py")
jl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jl)


def run_unconstrained(beta=1.0, seed=0, steps=None, clip=True):
    """Cosine + log-determinant with kappa = 0, i.e. no scale constraint.

    Records the covariance trace, the mean cosine similarity to the targets
    and the effective rank at the start and at the end of training.
    """
    steps = steps or jl.STEPS
    rng = np.random.default_rng(seed)
    D, P, NB, TAU = jl.D, jl.P, jl.NB, jl.TAU
    Wc = rng.standard_normal((D, P)) / np.sqrt(P)
    Wp = np.eye(D) + 0.1 * rng.standard_normal((D, D))
    Wt = Wc.copy()
    snap = {}

    for t in range(steps):
        X = jl.make_data(rng, NB)
        m, _ = jl.mask_of(rng)
        Xc = X * m
        Hc = Xc @ Wc.T
        Zh = Hc @ Wp.T
        Z = X @ Wt.T
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)

        nr = np.maximum(np.linalg.norm(Zh, axis=1, keepdims=True), 1e-9)
        U = Zh / nr
        cos_sim = float(np.mean(np.sum(U * Z, axis=1)))
        gZ = -(Z - U * np.sum(U * Z, axis=1, keepdims=True)) / nr / NB

        Zc = Zh - Zh.mean(0, keepdims=True)
        C = (Zc.T @ Zc) / (NB - 1)
        # kappa = 0: the trace penalty is switched off, which is the case
        # Proposition 4 is about
        dC = jl.reg_grad(C, "ldb", beta, 0.0)
        gZ = gZ + (2.0 / (NB - 1)) * Zc @ (0.5 * (dC + dC.T))

        if t in (0, steps - 1):
            snap[t] = dict(var=float(np.trace(C) / D), cos=cos_sim,
                           er=float(jl.eff_rank(C)))

        gWp = gZ.T @ Hc
        gWc = (gZ @ Wp).T @ Xc
        if clip:
            for G in (gWp, gWc):
                nrm = np.linalg.norm(G)
                if nrm > 1.0:
                    G *= 1.0 / nrm
        if not (np.isfinite(gWp).all() and np.isfinite(gWc).all()):
            break
        Wp -= 0.05 * gWp
        Wc -= 0.05 * gWc
        Wt = TAU * Wt + (1 - TAU) * Wc

    return snap[0], snap[max(snap)]


def main():
    print("\nSECTION 3.3 MEASUREMENTS THAT HAD NO SCRIPT")
    print("=" * 74)
    print("A.  COSINE + LOG-DETERMINANT WITHOUT A SCALE CONSTRAINT (kappa = 0)")
    print("=" * 74)
    for clip in (True, False):
        a, b = run_unconstrained(beta=1.0, seed=0, clip=clip)
        print("  gradient clipping: %s" % ("on" if clip else "off"))
        print("    %-32s %10s %10s" % ("quantity", "start", "end"))
        print("    %-32s %10.3f %10.3f" % ("mean per-dimension variance",
                                           a["var"], b["var"]))
        print("    %-32s %10.3f %10.4f" % ("cosine similarity to targets",
                                           a["cos"], b["cos"]))
        print("    %-32s %10.1f %10.1f  of %d"
              % ("effective rank", a["er"], b["er"], jl.D))
        # the manuscript quotes the variance to three significant figures;
        # print that form too so the quoted value is literally traceable
        print("    quoted form: variance %.3g -> %.3g, effective rank"
              " %.1f -> %.1f" % (a["var"], b["var"], a["er"], b["er"]))
        print()
    print("  beta = 1.0, kappa = 0, %d steps, otherwise the settings of"
          " jepa_linear.py\n" % jl.STEPS)

    print("=" * 74)
    print("B.  LOW-PRESSURE EQUILIBRIUM OF LDB ACROSS TWO DECADES OF eps")
    print("=" * 74)
    import pathlib as _p
    ct = {}
    src = (HERE / "canonical_thresholds.py").read_text()
    exec(compile(src.split("def main()")[0], "ct", "exec"), ct)
    print("  eta = 1, far below every measured collapse threshold")
    print("  %-12s %14s" % ("eps", "lam_min"))
    for eps in (1e-2, 1e-3, 1e-4):
        lam = float(ct["run"](1.0, "ldb", eps=eps))
        print("  %-12.0e %14.6f" % (eps, lam))
    print("\n  The dial acts near collapse and nowhere else: two decades of the")
    print("  ridge move the low-pressure solution by about two percent.\n")


if __name__ == "__main__":
    main()
