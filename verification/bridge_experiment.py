"""
BRIDGE EXPERIMENT
=================
Does Boundary-Limited Predictability (Theorem 1) survive on real language?

Theorem 1 is exact for stationary Markov sequences. Natural language is
neither stationary nor Markov, so the theorem cannot be applied directly.
What we test here is whether its *operational consequence* still holds.

Quantity measured
-----------------
For a character masked at depth d from the left edge of a span and depth r
from the right edge, the information any predictor could recover is

    i(d,r) = H(X_t) - H(X_t | X_{t-d}, X_{t+r})

For a span of length l, position p has (d,r) = (p, l+1-p), so

    S(l)    = sum_{p=1..l} i(p, l+1-p)        total recoverable info
    ibar(l) = S(l) / l                        per masked token

Theorem 1 predicts:  S(l) saturates,  ibar(l) = Theta(1/l).

Estimation
----------
Plug-in entropy is biased UPWARD, which would manufacture the effect we are
looking for. We therefore fit conditional distributions on a training half
and score cross-entropy on a held-out half. Held-out cross-entropy is an
upper bound on conditional entropy in expectation, so every i(d,r) reported
is a LOWER bound on the true recoverable information -- the conservative
direction. A shuffled-text control measures the residual bias floor.
"""

import numpy as np
from nltk.corpus import gutenberg

RNG = np.random.default_rng(0)
ALPHA = 0.5          # additive smoothing
MAXDEPTH = 20        # d, r range -> spans up to length 40
V = 27               # 26 letters + space


# ----------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------

def load_corpus():
    """11.8M chars of English -> 27-symbol alphabet (a-z + space)."""
    raw = " ".join(gutenberg.raw(f) for f in gutenberg.fileids()).lower()
    out = bytearray()
    prev_space = True
    for ch in raw:
        if "a" <= ch <= "z":
            out.append(ord(ch) - 97)
            prev_space = False
        elif not prev_space:
            out.append(26)          # collapse all whitespace/punct to one space
            prev_space = True
    return np.frombuffer(bytes(out), dtype=np.uint8)


# ----------------------------------------------------------------------
# Held-out conditional cross-entropy
# ----------------------------------------------------------------------

def cond_cross_entropy(train, test, d, r):
    """H(X_t | X_{t-d}, X_{t+r}) upper bound, in nats, scored on held-out data."""
    def ctx(x):
        left = x[:-(d + r)].astype(np.int32)         # X_{t-d}
        mid = x[d:len(x) - r].astype(np.int32)       # X_t
        right = x[d + r:].astype(np.int32)           # X_{t+r}
        return left * V + right, mid

    ctr, mtr = ctx(train)
    cte, mte = ctx(test)

    counts = np.bincount(ctr * V + mtr, minlength=V * V * V).reshape(V * V, V)
    probs = (counts + ALPHA) / (counts.sum(1, keepdims=True) + ALPHA * V)
    return float(-np.mean(np.log(probs[cte, mte])))


def marginal_cross_entropy(train, test):
    counts = np.bincount(train.astype(np.int32), minlength=V)
    probs = (counts + ALPHA) / (counts.sum() + ALPHA * V)
    return float(-np.mean(np.log(probs[test.astype(np.int32)])))


def build_info_table(train, test, label):
    """i(d,r) for all depths, in nats."""
    H0 = marginal_cross_entropy(train, test)
    I = np.zeros((MAXDEPTH + 1, MAXDEPTH + 1))
    for d in range(1, MAXDEPTH + 1):
        for r in range(1, MAXDEPTH + 1):
            I[d, r] = H0 - cond_cross_entropy(train, test, d, r)
    print(f"  [{label}] marginal H(X) = {H0:.4f} nats "
          f"({H0/np.log(2):.4f} bits)")
    return I, H0


# ----------------------------------------------------------------------
# Span aggregation
# ----------------------------------------------------------------------

def span_curves(I):
    """S(l) and ibar(l) for l = 1..2*MAXDEPTH-1."""
    S, ibar, ls = [], [], []
    for l in range(1, 2 * MAXDEPTH):
        tot = 0.0
        ok = True
        for p in range(1, l + 1):
            d, r = p, l + 1 - p
            if d > MAXDEPTH or r > MAXDEPTH:
                ok = False
                break
            tot += I[d, r]
        if not ok:
            continue
        ls.append(l)
        S.append(tot)
        ibar.append(tot / l)
    return np.array(ls), np.array(S), np.array(ibar)


# ----------------------------------------------------------------------

def main():
    print("\nBRIDGE EXPERIMENT -- Theorem 1 on real English text")
    print("=" * 74)

    x = load_corpus()
    half = len(x) // 2
    train, test = x[:half], x[half:]
    print(f"corpus: {len(x):,} symbols  |  train {len(train):,}  "
          f"test {len(test):,}\n")

    print("Estimating i(d,r) on real text ...")
    I, H0 = build_info_table(train, test, "real")

    print("\nEstimating i(d,r) on SHUFFLED text (bias floor control) ...")
    xs = x.copy()
    RNG.shuffle(xs)
    Is, _ = build_info_table(xs[:half], xs[half:], "shuffled")
    floor = np.abs(Is[1:, 1:]).max()
    print(f"  [shuffled] max |i(d,r)| = {floor:.2e} nats  <- bias floor")

    # ---------------- depth decay ----------------
    print("\n" + "=" * 74)
    print("A.  DECAY OF RECOVERABLE INFORMATION WITH DEPTH  (symmetric d=r)")
    print("=" * 74)
    print(f"  {'depth d=r':>10} {'i(d,r) nats':>13} {'% of i(1,1)':>13} "
          f"{'vs floor':>10}")
    print("  " + "-" * 50)
    base = I[1, 1]
    for d in [1, 2, 3, 4, 5, 6, 8, 10, 14, 20]:
        print(f"  {d:>10} {I[d,d]:>13.6f} {100*I[d,d]/base:>12.2f}% "
              f"{I[d,d]/floor:>9.0f}x")

    # ---------------- span curves ----------------
    ls, S, ibar = span_curves(I)
    print("\n" + "=" * 74)
    print("B.  SPAN-LEVEL CURVES:  does S(l) saturate and ibar(l) fall as 1/l ?")
    print("=" * 74)
    print(f"  {'span l':>7} {'S(l) nats':>11} {'% of S(39)':>11} "
          f"{'ibar(l)':>10} {'ibar*l':>9}")
    print("  " + "-" * 52)
    for l in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 39]:
        if l not in ls:
            continue
        j = int(np.where(ls == l)[0][0])
        print(f"  {l:>7} {S[j]:>11.6f} {100*S[j]/S[-1]:>10.2f}% "
              f"{ibar[j]:>10.6f} {ibar[j]*l:>9.4f}")

    slope = np.polyfit(np.log(ls[3:]), np.log(ibar[3:]), 1)[0]
    growth = np.polyfit(np.log(ls[3:]), S[3:], 1)[0]

    print("\n" + "=" * 74)
    print("C.  VERDICT")
    print("=" * 74)
    print(f"  log-log slope of ibar(l) vs l   = {slope:+.4f}   "
          f"(Markov theory: -1.0)")
    print(f"  S(l) fitted as a + b*log(l), b  = {growth:+.4f} nats/e-fold")
    print(f"  S(39) / S(4)                    = {S[-1]/S[3]:.3f}")

    strict = abs(slope + 1.0) < 0.15
    weak = slope < -0.6
    print()
    if strict:
        print("  => STRICT PASS: real English matches the Markov prediction")
        print("     closely. Theorem 1 transfers essentially unchanged.")
    elif weak:
        print("  => QUALIFIED PASS: per-token recoverable information still")
        print("     decays sharply with span length, but more slowly than")
        print("     Theta(1/l). Long-range dependence in language means S(l)")
        print("     keeps creeping up instead of fully saturating.")
        print("     ACTION: state Theorem 1 for the Markov idealisation, then")
        print("     report this measured exponent as the empirical analogue.")
        print("     Prediction P1 (interior optimum in l) SURVIVES either way,")
        print("     since it only needs ibar(l) to be decreasing.")
    else:
        print("  => FAIL: recoverable information does not decay with span")
        print("     length on real text. The framing must change before any")
        print("     compute is spent on training runs.")
    print("=" * 74)

    np.savez(str(__import__("pathlib").Path(__file__).parent / "bridge_results.npz"),
             I=I, ls=ls, S=S, ibar=ibar, floor=floor, H0=H0, slope=slope)


if __name__ == "__main__":
    main()
