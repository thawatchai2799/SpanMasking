"""
THE CENTRAL COMPARISON, RUN PROPERLY FOR THE FIRST TIME
=======================================================
Every VICReg-vs-LDB number in this project came from the regulariser alone
under an artificial pressure eta*lam_min, with fixed random targets and no
momentum encoder. That is not the setting the paper is about. Collapse in JEPA
arises because the target itself is produced by an EMA copy of the encoder, so
the whole system can drift to a trivial solution -- a mechanism absent from
every experiment run so far.

This builds a minimal but genuine linear JEPA:

    x        ~ N(0, Sigma)          Sigma = AR(1) over p coordinates
    x_c      = masked copy of x     (contiguous span)
    z_hat    = W_p W_c x_c          context encoder + predictor
    z        = sg(W_t x)            target encoder, EMA of W_c
    loss     = cosine(z_hat, z) + regulariser(Cov(z_hat))
    W_t      <- tau W_t + (1-tau) W_c

and evaluates representations the way the paper proposes to: a linear probe
from z_hat to the MASKED coordinates, scored as R^2 on held-out data. That is
a real downstream task, not a proxy for the objective.

Questions:
  Q1  Does cosine-only actually collapse here? If not, the regulariser
      question is moot in this setting and we need to say so.
  Q2  With alignment present, does LDB beat VICReg on the probe?
  Q3  Is the trace-penalty coefficient kappa a free lunch or a tuned knob?
  Q4  Does Theorem 2's weight-decay law survive the trace penalty?
"""

import numpy as np

P, D, NB, STEPS, TAU = 64, 32, 256, 3000, 0.99
EPS = 1e-3


def ar1(p, r=0.9):
    i = np.arange(p)
    return r ** np.abs(i[:, None] - i[None, :])


def make_data(rng, n):
    L = np.linalg.cholesky(ar1(P) + 1e-6 * np.eye(P))
    return rng.standard_normal((n, P)) @ L.T


def mask_of(rng, span=16):
    s = rng.integers(0, P - span)
    m = np.ones(P)
    m[s:s + span] = 0.0
    return m, np.arange(s, s + span)


def eff_rank(C):
    w = np.maximum(np.linalg.eigvalsh(C), 0)
    p = w / max(w.sum(), 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def reg_grad(C, mode, beta, kappa, mu=25.0, nu=1.0, gam=1.0):
    d = C.shape[0]
    if mode == "none":
        return np.zeros_like(C)
    if mode == "vicreg":
        var = np.sqrt(np.maximum(np.diag(C), 0) + 1e-8)
        h = np.maximum(0.0, gam - var)
        g = np.zeros_like(C)
        np.fill_diagonal(g, mu * (-2.0 * h * (0.5 / var)))
        return g + nu * 2.0 * (C - np.diag(np.diag(C)))
    g = -(beta / 2.0) * np.linalg.inv(C + EPS * np.eye(d))
    g = g + (kappa * (np.trace(C) / d - 1.0) / d) * np.eye(d)
    return g


def train(mode, beta=0.01, kappa=1.0, omega=0.0, seed=0, mu=25.0, nu=1.0):
    rng = np.random.default_rng(seed)
    Wc = rng.standard_normal((D, P)) / np.sqrt(P)
    Wp = np.eye(D) + 0.1 * rng.standard_normal((D, D))
    Wt = Wc.copy()
    wnorms = []
    for t in range(STEPS):
        X = make_data(rng, NB)
        m, _ = mask_of(rng)
        Xc = X * m
        Hc = Xc @ Wc.T
        Zh = Hc @ Wp.T
        Z = X @ Wt.T
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)

        nr = np.maximum(np.linalg.norm(Zh, axis=1, keepdims=True), 1e-9)
        U = Zh / nr
        gZ = -(Z - U * np.sum(U * Z, axis=1, keepdims=True)) / nr / NB

        Zc = Zh - Zh.mean(0, keepdims=True)
        C = (Zc.T @ Zc) / (NB - 1)
        dC = reg_grad(C, mode, beta, kappa, mu, nu)
        if mode != "none":
            gZ = gZ + (2.0 / (NB - 1)) * Zc @ (0.5 * (dC + dC.T))

        gWp = gZ.T @ Hc + omega * Wp
        gWc = (gZ @ Wp).T @ Xc + omega * Wc
        # global-norm clipping, as in ordinary training; without it the
        # heavier regularisers diverge at this step size and the comparison
        # would be a comparison of learning rates
        for G in (gWp, gWc):
            nrm = np.linalg.norm(G)
            if nrm > 1.0:
                G *= 1.0 / nrm
        if not (np.isfinite(gWp).all() and np.isfinite(gWc).all()):
            break
        Wp -= 0.05 * gWp
        Wc -= 0.05 * gWc
        Wt = TAU * Wt + (1 - TAU) * Wc
        if t % 200 == 0:
            wnorms.append(np.linalg.norm(Wp))

    # ---- evaluation: linear probe from z_hat to the masked coordinates
    rng2 = np.random.default_rng(seed + 999)
    Xtr, Xte = make_data(rng2, 2000), make_data(rng2, 2000)
    m, idx = mask_of(rng2)
    f = lambda A: ((A * m) @ Wc.T) @ Wp.T
    Ztr, Zte = f(Xtr), f(Xte)
    Ztr = np.hstack([Ztr, np.ones((len(Ztr), 1))])
    Zte = np.hstack([Zte, np.ones((len(Zte), 1))])
    Ytr, Yte = Xtr[:, idx], Xte[:, idx]
    if not np.isfinite(Ztr).all():
        return dict(r2=float('nan'), er=float('nan'), lam=float('nan'),
                    tr=float('nan'), wn=[1.0])
    Wl = np.linalg.lstsq(Ztr, Ytr, rcond=1e-8)[0]
    pred = Zte @ Wl
    r2 = 1.0 - ((Yte - pred) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()

    Zc = Ztr[:, :-1] - Ztr[:, :-1].mean(0, keepdims=True)
    C = (Zc.T @ Zc) / (len(Zc) - 1)
    return dict(r2=float(r2), er=eff_rank(C),
                lam=float(np.maximum(np.linalg.eigvalsh(C), 0)[0]),
                tr=float(np.trace(C) / D), wn=wnorms)


def main():
    print("\nLINEAR JEPA WITH EMA TARGETS -- THE CENTRAL COMPARISON")
    print("=" * 78)
    print(f"  p={P}, d={D}, batch={NB}, {STEPS} steps, tau={TAU}, span=16")
    print("  probe: linear map from z_hat to the masked coordinates, "
          "held-out R^2\n")
    seeds = (0, 1, 2)

    def summarise(label, **kw):
        rs = [train(seed=s, **kw) for s in seeds]
        f = lambda k: (np.mean([r[k] for r in rs]), np.std([r[k] for r in rs]))
        r2m, r2s = f("r2"); erm, _ = f("er"); lm, _ = f("lam"); tr, _ = f("tr")
        print(f"  {label:<26} {r2m:>7.4f} +/-{r2s:<6.4f} {erm:>9.2f}"
              f" {lm:>11.3e} {tr:>9.3f}")
        return r2m

    print(f"  {'configuration':<26} {'probe R^2':>15} {'eff.rank':>9}"
          f" {'lam_min':>11} {'tr(C)/d':>9}")
    print("  " + "-" * 74)
    base = summarise("cosine only", mode="none")
    summarise("+ VICReg (25, 1)", mode="vicreg")
    summarise("+ VICReg (25, 16)", mode="vicreg", nu=16.0)
    summarise("+ LDB beta=0.01", mode="ldb", beta=0.01, kappa=1.0)
    summarise("+ LDB beta=0.1", mode="ldb", beta=0.1, kappa=1.0)
    summarise("+ LDB beta=1.0", mode="ldb", beta=1.0, kappa=1.0)

    print("\n  Q3: is kappa a free knob or a tuned one?")
    print(f"  {'kappa (beta=0.01)':<26} {'probe R^2':>15} {'eff.rank':>9}"
          f" {'lam_min':>11} {'tr(C)/d':>9}")
    print("  " + "-" * 74)
    for kap in (0.1, 1.0, 10.0):
        summarise(f"  kappa={kap}", mode="ldb", beta=0.01, kappa=kap)

    print("\n  Q4: does Theorem 2's weight-decay law survive the trace penalty?")
    for mode, lab in (("none", "cosine only"), ("ldb", "cosine + LDB")):
        r = train(mode=mode, beta=0.01, kappa=1.0, omega=0.2, seed=0)
        w = np.array(r["wn"])
        t = np.arange(len(w)) * 200 * 0.05
        rate = -np.polyfit(t[: len(t) // 2], np.log(w[: len(t) // 2]), 1)[0]
        print(f"    {lab:<16} fitted decay rate = {rate:+.4f}"
              f"   (omega = 0.2)")
    print("=" * 78)


if __name__ == "__main__":
    main()
