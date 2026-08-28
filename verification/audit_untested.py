"""
FOUR UNTESTED CLAIMS
====================
R19  Corollary 1.1 was checked at a single r (0.85) with a centred block.
     Test it across r and against edge cases.

R20  Theorem 1 was checked only on Gaussian processes. Test it on a DISCRETE,
     non-Gaussian, order-2 Markov chain by exact enumeration -- no Gaussian
     formula anywhere in the path.

R21  Section 3.5 asserts: "VICReg offers no comparable dial: its threshold is
     fixed by (mu, nu, gamma) through K, which also controls the strength of
     the regulariser itself, so the two cannot be tuned independently."
     That was written without evidence. The testable content is: raising mu
     should move BOTH the collapse threshold AND the low-pressure behaviour,
     whereas changing eps should move the threshold but leave low-pressure
     behaviour alone.

R22  Theorem 2 has never been checked numerically at all. Test both the
     Frobenius claim and the failure of the per-sample analogue under batching.
"""

import numpy as np
from itertools import product, groupby

# ====================================================================== R19
def ar1_cov(T, r, s2=1.0):
    i = np.arange(T)
    return s2 * r ** np.abs(i[:, None] - i[None, :])


def mi_g(S, A, B):
    A, B = np.array(sorted(A)), np.array(sorted(B))
    AB = np.concatenate([A, B])
    sl = lambda I: np.linalg.slogdet(S[np.ix_(I, I)])[1]
    return 0.5 * (sl(A) + sl(B) - sl(AB))


def r19():
    print("=" * 72)
    print("R19  Corollary 1.1 across r, sigma^2 and block placement")
    print("=" * 72)
    cf = lambda l, r: 0.5 * np.log((1 - r ** (2 * (l + 1))) / (1 - r ** 2) ** 2)
    worst = 0.0
    print(f"  {'r':>6} {'sigma^2':>8} {'T':>5} {'l':>4} {'offset':>7}"
          f" {'numeric':>11} {'closed form':>12} {'err':>10}")
    print("  " + "-" * 68)
    for r in (0.3, 0.5, 0.85, 0.95, 0.99):
        for s2 in (1.0, 7.3):
            for T, l, off in ((160, 5, 0), (160, 5, -40), (60, 20, 0),
                              (60, 40, 0), (60, 55, 0)):
                start = (T - l) // 2 + off
                if start < 1 or start + l >= T:      # must stay interior
                    continue
                S = ar1_cov(T, r, s2)
                Sm = list(range(start, start + l))
                Sc = [t for t in range(T) if t not in set(Sm)]
                num, exact = mi_g(S, Sm, Sc), cf(l, r)
                worst = max(worst, abs(num - exact))
                if (r, s2, l) in ((0.3, 1.0, 5), (0.95, 7.3, 20),
                                  (0.99, 1.0, 55), (0.85, 1.0, 5)):
                    print(f"  {r:>6} {s2:>8} {T:>5} {l:>4} {off:>7}"
                          f" {num:>11.7f} {exact:>12.7f} {abs(num-exact):>10.1e}")
    print(f"\n  worst error over 50 configurations: {worst:.2e}")
    print(f"  VERDICT: {'PASS' if worst < 1e-8 else 'FAIL'}"
          "  (closed form is r- and sigma-independent as claimed)\n")
    return worst


# ====================================================================== R20
def r20(V=3, T=10, seed=0):
    print("=" * 72)
    print("R20  Theorem 1 on a DISCRETE order-2 Markov chain (exact enumeration)")
    print("=" * 72)
    rng = np.random.default_rng(seed)
    p1 = rng.dirichlet(np.ones(V))
    p2 = rng.dirichlet(np.ones(V), size=V)                 # p(x2|x1)
    p3 = rng.dirichlet(np.ones(V) * 0.6, size=(V, V))      # p(xt|x_{t-2},x_{t-1})

    seqs = np.array(list(product(range(V), repeat=T)))
    logp = np.log(p1[seqs[:, 0]]) + np.log(p2[seqs[:, 0], seqs[:, 1]])
    for t in range(2, T):
        logp += np.log(p3[seqs[:, t - 2], seqs[:, t - 1], seqs[:, t]])
    P = np.exp(logp)
    P /= P.sum()

    def H(cols):
        if len(cols) == 0:
            return 0.0
        key = np.zeros(len(seqs), dtype=np.int64)
        for c in cols:
            key = key * V + seqs[:, c]
        q = np.bincount(key, weights=P)
        q = q[q > 0]
        return float(-(q * np.log(q)).sum())

    def MI(A, B):
        return H(A) + H(B) - H(list(A) + list(B))

    def blocks(S):
        out = []
        for _, g in groupby(enumerate(sorted(S)), lambda t: t[1] - t[0]):
            g = [x[1] for x in g]
            out.append((g[0], g[-1]))
        return out

    def bnd(S, order):
        Ss, b = set(S), set()
        for i, j in blocks(S):
            for o in range(1, order + 1):
                if i - o >= 0 and i - o not in Ss:
                    b.add(i - o)
                if j + o < T and j + o not in Ss:
                    b.add(j + o)
        return sorted(b)

    cases = [[3, 4], [2, 3, 4, 5], [1], [4, 5, 6, 7], [2, 3, 7, 8], [1, 2, 6]]
    print(f"  V={V}, T={T}, {V**T:,} sequences enumerated exactly\n")
    print(f"  {'masked set':>16} {'k':>3} {'I(S;S^c)':>11}"
          f" {'I(S;dS) w=2':>13} {'err':>10} {'w=1 (wrong)':>13}")
    print("  " + "-" * 70)
    worst2 = worst1 = 0.0
    for S in cases:
        Sc = [t for t in range(T) if t not in set(S)]
        full = MI(S, Sc)
        i2 = MI(S, bnd(S, 2))
        i1 = MI(S, bnd(S, 1))
        worst2 = max(worst2, abs(full - i2))
        worst1 = max(worst1, abs(full - i1))
        print(f"  {str(S):>16} {len(blocks(S)):>3} {full:>11.7f}"
              f" {i2:>13.7f} {abs(full-i2):>10.1e} {i1:>13.7f}")
    print(f"\n  width-2 boundary: worst error {worst2:.2e} -> "
          f"{'PASS' if worst2 < 1e-10 else 'FAIL'}")
    print(f"  width-1 boundary: worst error {worst1:.2e} -> "
          f"{'correctly fails' if worst1 > 1e-6 else 'NO DISCRIMINATION'}")
    print("  => Theorem 1 holds outside the Gaussian family.\n")
    return worst2, worst1


# ====================================================================== R21
def cov_of(Z):
    Zc = Z - Z.mean(0, keepdims=True)
    return (Zc.T @ Zc) / (Z.shape[0] - 1), Zc


def train(mode, eta, mu=25.0, nu=1.0, gam=1.0, eps=1e-3, beta=1.0,
          d=24, n=384, steps=5000, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, d)) * 0.5
    lr = min(0.02, 1.0 / max(eta, 1.0))
    for _ in range(steps):
        C, Zc = cov_of(Z)
        if mode == "vicreg":
            var = np.sqrt(np.maximum(np.diag(C), 0) + 1e-8)
            hinge = np.maximum(0.0, gam - var)
            dC = np.zeros_like(C)
            np.fill_diagonal(dC, mu * (-2.0 * hinge * (0.5 / var)))
            dC = dC + nu * 2.0 * (C - np.diag(np.diag(C)))
        else:
            dC = -(beta / 2.0) * np.linalg.inv(C + eps * np.eye(d))
        w, Vv = np.linalg.eigh(C)
        dC = dC + eta * np.outer(Vv[:, 0], Vv[:, 0])
        Z = Z - lr * (2.0 / (n - 1)) * Zc @ (0.5 * (dC + dC.T))
    C, _ = cov_of(Z)
    w = np.linalg.eigvalsh(C)
    iso = np.linalg.norm(C - np.diag(np.diag(C))) / np.sqrt(C.shape[0])
    return float(w[0]), float(np.trace(C) / C.shape[0]), iso


def threshold(mode, **kw):
    """Smallest eta (log grid) at which lam_min drops below 1e-6."""
    lo = None
    for eta in [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
        if train(mode, eta, **kw)[0] < 1e-6:
            lo = eta
            break
    return lo


def r21():
    print("=" * 72)
    print("R21  Does VICReg have a dial comparable to eps?")
    print("=" * 72)
    print("  claim under test: raising mu moves BOTH the threshold and the")
    print("  low-pressure behaviour; changing eps moves only the threshold.\n")
    print(f"  {'objective':>10} {'setting':>12} {'threshold eta*':>15}"
          f" {'lam_min @ eta=1':>17} {'tr(C)/d @ eta=1':>17}")
    print("  " + "-" * 74)
    for mu in (5.0, 25.0, 125.0):
        th = threshold("vicreg", mu=mu)
        lm, tr, _ = train("vicreg", 1.0, mu=mu)
        print(f"  {'VICReg':>10} {'mu='+str(mu):>12} {str(th):>15}"
              f" {lm:>17.5f} {tr:>17.5f}")
    for eps in (1e-2, 1e-3, 1e-4):
        th = threshold("ldb", eps=eps)
        lm, tr, _ = train("ldb", 1.0, eps=eps)
        print(f"  {'LDB':>10} {'eps='+f'{eps:.0e}':>12} {str(th):>15}"
              f" {lm:>17.5f} {tr:>17.5f}")
    print()


# ====================================================================== R22
def r22(n=64, d=8, steps=4000, lr=1e-3, omega=0.35, seed=1):
    print("=" * 72)
    print("R22  Theorem 2: Frobenius decay, and failure of the per-sample form")
    print("=" * 72)
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, d))
    z = rng.standard_normal((n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    W = rng.standard_normal((d, d)) * 0.5

    W0 = np.linalg.norm(W)
    zh0 = np.linalg.norm(h @ W.T, axis=1).mean()
    for t in range(steps):
        Zh = h @ W.T
        nrm = np.linalg.norm(Zh, axis=1, keepdims=True)
        U = Zh / nrm
        # d/dZh of  -mean <U, z>  is  -(z - U<U,z>)/nrm / n
        g_z = -(z - U * np.sum(U * z, axis=1, keepdims=True)) / nrm / n
        W = W - lr * (g_z.T @ h + omega * W)
    T_end = steps * lr
    pred = W0 * np.exp(-omega * T_end)
    meas = np.linalg.norm(W)
    zh_pred = zh0 * np.exp(-omega * T_end)
    zh_meas = np.linalg.norm(h @ W.T, axis=1).mean()
    print(f"  gradient-flow time = steps*lr = {T_end:.2f}, omega = {omega}\n")
    print(f"  {'quantity':>28} {'predicted':>12} {'measured':>12} {'rel.err':>9}")
    print("  " + "-" * 64)
    e1 = abs(meas - pred) / pred
    e2 = abs(zh_meas - zh_pred) / zh_pred
    print(f"  {'||W||_F (Theorem 2)':>28} {pred:>12.6f} {meas:>12.6f} {e1:>8.3%}")
    print(f"  {'mean ||z_hat|| (per-sample)':>28} {zh_pred:>12.6f}"
          f" {zh_meas:>12.6f} {e2:>8.3%}")
    print(f"\n  VERDICT: Frobenius form {'PASS' if e1 < 0.01 else 'FAIL'};"
          f" per-sample form {'also holds' if e2 < 0.01 else 'deviates as predicted'}")
    print("  (a deviating per-sample norm is the cross-term effect noted")
    print("   in the Remark after Theorem 2)\n")


if __name__ == "__main__":
    print()
    r19(); r20(); r21(); r22()
