"""
EXPERIMENT DESIGN FOR P1, WITH THE EFFECT SIZE COMPUTED IN ADVANCE
=================================================================
P1 predicts that representation quality tracks the block count k rather than
the masking ratio rho. The 2x2 design needs configurations where those two
come apart far enough to be measurable, and we should know the expected
separation BEFORE spending compute, not after.

For each candidate (rho, l) we compute I(X_S ; X_{S^c}) exactly under a
Gaussian AR(1) surrogate at the sequence length the models will use. The AR(1)
correlation is set from the measured decay of recoverable information in
English (Section 3.5): for AR(1) the information a
neighbour at distance d carries is -0.5*log(1 - r^(2d)), whose log decays at
kappa = -2 log r, so a measured kappa ~ 0.62 inverts to r = exp(-kappa/2)
~ 0.73, NOT exp(-kappa). Even that holds only in the small-information regime,
which is a further reason not to lean on the calibration. We report the design under a
range of r so the conclusion does not rest on that calibration.

The design is falsifiable in a specific way: if the predicted within-pair
information gaps are similar for the matched-k and matched-rho pairs, the
experiment cannot discriminate and P1 should be dropped rather than run.

GROUNDING IN MEASURED ENGLISH (computed separately from the bridge tables,
interleaved split, k=2 context): per-cell diagnostic sums give a matched-k gap
of 0.18 and a matched-rho gap of 0.59, a discrimination ratio of 3.2x -- lower
than the AR(1) surrogate suggests (4.0-43.6x) but with the predicted ordering
intact. The surrogate overstates the design's power and 3.2x is the number to
plan around.
"""

import numpy as np
from itertools import groupby

T = 128            # model sequence length


def ar1(T, r):
    i = np.arange(T)
    return r ** np.abs(i[:, None] - i[None, :])


def mi(S, A, B):
    A, B = np.array(sorted(A)), np.array(sorted(B))
    AB = np.concatenate([A, B])
    sl = lambda I: np.linalg.slogdet(S[np.ix_(I, I)])[1]
    return 0.5 * (sl(A) + sl(B) - sl(AB))


def place(T, k, l, margin=2):
    """k blocks of length l, evenly spaced, no merging."""
    stride = (T - 2 * margin) / k
    if stride < l + 1:
        return None                      # blocks would touch
    S = []
    for b in range(k):
        s = int(round(margin + b * stride))
        s = min(s, T - margin - l)
        S.extend(range(s, s + l))
    S = sorted(set(S))
    nb = len(list(groupby(enumerate(S), lambda t: t[1] - t[0])))
    return S if nb == k else None


def info(Sig, S):
    Sc = [t for t in range(T) if t not in set(S)]
    return mi(Sig, S, Sc)


def main():
    print("\nDESIGN FOR P1: matched-k versus matched-rho")
    print("=" * 78)
    print(f"  sequence length T = {T}\n")

    # Three distinct cells forming an L, not a full 2x2: B is the shared
    # corner of both comparisons and is trained once.
    #
    # STRUCTURAL CONFOUND. k = m/l, so matching k and m simultaneously forces
    # the same l -- the same configuration. The matched-k pair therefore
    # differs in m (16 vs 32), i.e. in the number of positions predicted per
    # step, which is the training-signal side of the trade-off Wettig et al.
    # document. The bias runs against us: less signal should make cell A
    # WORSE, while P1 predicts A ~ B, so the confound can only cause a false
    # rejection, never a false confirmation. Mitigation is to equalise total
    # masked-token predictions by giving cell A twice the steps (or batch),
    # and to report equalised and unequalised runs separately.
    cells = {
        "A": dict(k=8, l=2),      # rho = 0.125
        "B": dict(k=8, l=4),      # rho = 0.250   (shared corner)
        "D": dict(k=4, l=8),      # rho = 0.250
    }
    print("  three distinct cells in an L, not a full 2x2; B is the shared")
    print("  corner of both comparisons and is trained once:\n")
    print(f"  {'cell':>5} {'k':>4} {'l':>4} {'|S|':>5} {'rho':>7}")
    print("  " + "-" * 30)
    S_of = {}
    for name, c in cells.items():
        S = place(T, c["k"], c["l"])
        if S is None:
            print(f"  {name:>5}  INFEASIBLE at T={T}")
            continue
        S_of[name] = S
        print(f"  {name:>5} {c['k']:>4} {c['l']:>4} {len(S):>5} {len(S)/T:>7.3f}")

    print("\n  predicted information under AR(1) surrogates:\n")
    print(f"  {'r':>6} {'I(A)':>8} {'I(B)':>8} {'I(D)':>8}"
          f" {'matched-k gap':>15} {'matched-rho gap':>17} {'ratio':>8}")
    print("  " + "-" * 74)
    rows = []
    for r in (0.40, 0.55, 0.73, 0.85, 0.95):   # 0.73 is the calibrated value
        Sig = ar1(T, r)
        IA, IB, ID = (info(Sig, S_of[x]) for x in ("A", "B", "D"))
        gk = abs(IA - IB) / max(IA, IB)          # same k, different rho
        gr = abs(IB - ID) / max(IB, ID)          # same rho, different k
        ratio = gr / max(gk, 1e-12)
        rows.append((r, gk, gr, ratio))
        print(f"  {r:>6.2f} {IA:>8.4f} {IB:>8.4f} {ID:>8.4f}"
              f" {gk:>15.4f} {gr:>17.4f} {ratio:>8.1f}x")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    worst = min(x[3] for x in rows)
    best = max(x[3] for x in rows)
    print(f"  discrimination ratio across r in [0.40, 0.95]: "
          f"{worst:.1f}x to {best:.1f}x")
    small = min(x[1] for x in rows)
    print(f"  smallest matched-k gap to be resolved: {small:.3f} "
          f"relative information")
    if worst > 3:
        print("\n  => The design discriminates across the whole calibration")
        print("     range, so the conclusion does not depend on pinning r.")
        print("     P1 is worth running.")
    else:
        print("\n  => The design fails to separate the two pairs at some r.")
        print("     P1 should be dropped rather than run.")

    print("\n  RUN PLAN")
    print("  " + "-" * 74)
    n_cells, seeds = 3, 5
    base = n_cells * seeds
    print(f"  cells to train: {n_cells} (A, B, D)   seeds: {seeds}"
          f"   base runs: {base}")
    print(f"  one objective (cosine) for P1; P4 adds an MLM arm at cell B")
    print(f"    (B + auxiliary MLM) x {seeds} seeds = {seeds} runs")
    print(f"  cell A at 2x steps (signal equalisation) x {seeds} = {seeds} runs")
    print(f"    NOTE: 2x steps also means 2x data seen; equalisation trades one")
    print(f"    confound for another and both versions are reported for that reason")
    print(f"  TOTAL runs: {base + 2*seeds}")
    print(f"  P1b (unimodality in l) CANNOT reuse these runs: the cells differ")
    print(f"    in rho and k as well as l. A valid P1b sweep needs l varied at")
    print(f"    fixed rho, i.e. extra cells; we run it only if budget remains")
    print(f"    and otherwise leave P1b as stated-not-tested, as the draft says.")
    print(f"  diagnostics per run: probe accuracy, effective rank, lam_min,")
    print(f"    ||W||_F trajectory WITHOUT gradient clipping in the P3 runs")
    print("=" * 78)


if __name__ == "__main__":
    main()
