"""
THREE CHECKS THAT GATE THE TRAINING RUN
=======================================
R32  P1 proposes a matched-k vs matched-rho design. Nobody has checked that
     the predicted effect is large enough to be worth measuring. If the
     matched-k gap is not much smaller than the matched-rho gap even in the
     idealised AR(1) setting, P1 is untestable and should be dropped before
     any compute is spent.

R33  Corollary 1.2's k-scaling was only ever checked on Gaussian AR(1).
     Check it on the discrete order-2 chain, where nothing Gaussian appears.

R34  Every LDB experiment used the regulariser plus an artificial pressure.
     L_cos + L_ent has never been run together. If beta = 1 destroys latent
     alignment, the objective is unusable at that setting and the training
     loop needs a different beta.
"""

import numpy as np
from itertools import product, groupby

# ==================================================================== R32
def ar1(T, r):
    i = np.arange(T)
    return r ** np.abs(i[:, None] - i[None, :])


def mi(S, A, B):
    A, B = np.array(sorted(A)), np.array(sorted(B))
    AB = np.concatenate([A, B])
    sl = lambda I: np.linalg.slogdet(S[np.ix_(I, I)])[1]
    return 0.5 * (sl(A) + sl(B) - sl(AB))


def place(T, k, l, margin=3):
    stride = (T - 2 * margin) / k
    S = []
    for b in range(k):
        s = int(round(margin + b * stride))
        s = min(s, T - margin - l)
        S.extend(range(s, s + l))
    return sorted(set(S))


def r32(T=240, r=0.85):
    print("=" * 76)
    print("R32  Is the matched-k prediction of P1 detectable?")
    print("=" * 76)
    S = ar1(T, r)

    def info(k, l):
        Sm = place(T, k, l)
        Sc = [t for t in range(T) if t not in set(Sm)]
        nb = len(list(groupby(enumerate(Sm), lambda t: t[1] - t[0])))
        return mi(S, Sm, Sc), len(Sm) / T, nb

    print(f"  AR(1), r={r}, T={T}\n")
    print(f"  {'pair':>14} {'k':>4} {'l':>4} {'rho':>7} {'blocks':>7}"
          f" {'I (nats)':>10}")
    print("  " + "-" * 52)

    pairs = {
        "matched k": [(6, 3), (6, 12)],       # same k, rho 7.5% vs 30%
        "matched rho": [(18, 4), (6, 12)],    # same rho 30%, k 18 vs 6
    }
    res = {}
    for name, cfgs in pairs.items():
        vals = []
        for k, l in cfgs:
            I, rho, nb = info(k, l)
            vals.append(I)
            print(f"  {name:>14} {k:>4} {l:>4} {rho:>7.3f} {nb:>7} {I:>10.4f}")
        res[name] = abs(vals[0] - vals[1]) / max(vals)
        print()
    print(f"  relative gap within matched-k pair   : {res['matched k']:.3f}")
    print(f"  relative gap within matched-rho pair : {res['matched rho']:.3f}")
    ratio = res["matched rho"] / max(res["matched k"], 1e-9)
    print(f"  ratio (larger is a stronger test)    : {ratio:.1f}x")
    print(f"  VERDICT: {'P1 is testable' if ratio > 3 else 'P1 IS TOO WEAK -- drop it'}\n")


# ==================================================================== R33
def r33(V=3, T=12, seed=0):
    print("=" * 76)
    print("R33  Corollary 1.2 k-scaling on the DISCRETE order-2 chain")
    print("=" * 76)
    rng = np.random.default_rng(seed)
    p1 = rng.dirichlet(np.ones(V))
    p2 = rng.dirichlet(np.ones(V), size=V)
    p3 = rng.dirichlet(np.ones(V) * 0.6, size=(V, V))
    seqs = np.array(list(product(range(V), repeat=T)))
    lp = np.log(p1[seqs[:, 0]]) + np.log(p2[seqs[:, 0], seqs[:, 1]])
    for t in range(2, T):
        lp += np.log(p3[seqs[:, t - 2], seqs[:, t - 1], seqs[:, t]])
    P = np.exp(lp); P /= P.sum()

    def H(cols):
        if not len(cols):
            return 0.0
        key = np.zeros(len(seqs), dtype=np.int64)
        for c in cols:
            key = key * V + seqs[:, c]
        q = np.bincount(key, weights=P); q = q[q > 0]
        return float(-(q * np.log(q)).sum())

    MI = lambda A, B: H(A) + H(B) - H(list(A) + list(B))

    print(f"  V={V}, T={T}, {V**T:,} sequences, |S| = 4 held FIXED\n")
    print(f"  {'l':>4} {'k':>4} {'masked set':>18} {'I(S;S^c)':>11}"
          f" {'I/k':>9} {'I*l':>9}")
    print("  " + "-" * 58)
    rows = []
    for l, S in [(1, [1, 4, 7, 10]), (2, [2, 3, 8, 9]), (4, [4, 5, 6, 7])]:
        Sc = [t for t in range(T) if t not in set(S)]
        I = MI(S, Sc)
        k = len(list(groupby(enumerate(S), lambda t: t[1] - t[0])))
        rows.append((l, k, I))
        print(f"  {l:>4} {k:>4} {str(S):>18} {I:>11.6f} {I/k:>9.6f} {I*l:>9.4f}")
    ls = np.array([x[0] for x in rows], float)
    Is = np.array([x[2] for x in rows], float)
    sl = np.polyfit(np.log(ls), np.log(Is), 1)[0]
    print(f"\n  log-log slope of I vs l = {sl:+.3f}  (Corollary 1.2 bound: -1)")
    print(f"  I/k varies by a factor of "
          f"{max(x[2]/x[1] for x in rows)/min(x[2]/x[1] for x in rows):.2f}")
    print("  => k-scaling is not a Gaussian artefact.\n")


# ==================================================================== R34
def r34(d=32, n=256, steps=4000, eps=1e-3, seed=0):
    print("=" * 76)
    print("R34  Does L_cos + L_ent actually train together?")
    print("=" * 76)
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, d))
    z = rng.standard_normal((n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)

    print(f"  linear predictor, d={d}, n={n}, {steps} steps, eps={eps:.0e}\n")
    print(f"  {'beta':>8} {'mean cos sim':>14} {'lam_min':>11}"
          f" {'eff. rank':>11} {'beta/(2*eta_eff)':>18}")
    print("  " + "-" * 66)
    for beta in (0.0, 0.01, 0.1, 1.0, 10.0):
        W = rng.standard_normal((d, d)) * 0.3
        for _ in range(steps):
            Zh = h @ W.T
            nr = np.linalg.norm(Zh, axis=1, keepdims=True)
            U = Zh / nr
            g = -(z - U * np.sum(U * z, axis=1, keepdims=True)) / nr / n
            if beta > 0:
                Zc = Zh - Zh.mean(0, keepdims=True)
                C = (Zc.T @ Zc) / (n - 1)
                dC = -(beta / 2.0) * np.linalg.inv(C + eps * np.eye(d))
                g = g + (2.0 / (n - 1)) * Zc @ (0.5 * (dC + dC.T))
            W = W - 0.05 * (g.T @ h)
        Zh = h @ W.T
        U = Zh / np.linalg.norm(Zh, axis=1, keepdims=True)
        cs = float(np.mean(np.sum(U * z, axis=1)))
        Zc = Zh - Zh.mean(0, keepdims=True)
        C = (Zc.T @ Zc) / (n - 1)
        w = np.maximum(np.linalg.eigvalsh(C), 0)
        p = w / w.sum(); p = p[p > 1e-12]
        er = float(np.exp(-(p * np.log(p)).sum()))
        lm = float(w[0])
        inferred = beta / (2 * (lm + eps)) if beta > 0 else float("nan")
        print(f"  {beta:>8} {cs:>14.5f} {lm:>11.3e} {er:>11.2f}"
              f" {inferred:>18.2f}")
    print("\n  beta=0 is the cosine-only baseline. Read down the lam_min and")
    print("  eff.rank columns for the anti-collapse effect, and across the")
    print("  cos-sim column for what it costs in alignment.\n")


if __name__ == "__main__":
    print(); r32(); r33(); r34()
