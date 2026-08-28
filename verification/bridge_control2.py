"""
CONTEXT-WIDTH CONTROL (exact, k = 1 vs k = 2)
=============================================
The k=3 attempt is dropped: 27^6 = 387M contexts cannot be estimated by
counting from 5.5M training symbols, and hashing into 1M buckets destroyed
so much information that cross-entropy went back up to the k=1 level. That
is a limitation of plug-in counting, not a result, so it is not reported as
one.

What remains is an exact comparison of k=1 (27^2 = 729 contexts) against
k=2 (27^4 = 531,441 contexts). If the decay exponent of ibar(l) is stable
across a 729x increase in conditioning power, the 1/l scaling is a property
of the language rather than of the estimator's Markov assumption.
"""

import numpy as np
from nltk.corpus import gutenberg

ALPHA = 0.5
V = 27
MAXD = 7


def load_corpus():
    raw = " ".join(gutenberg.raw(f) for f in gutenberg.fileids()).lower()
    out, prev_space = bytearray(), True
    for ch in raw:
        if "a" <= ch <= "z":
            out.append(ord(ch) - 97); prev_space = False
        elif not prev_space:
            out.append(26); prev_space = True
    return np.frombuffer(bytes(out), dtype=np.uint8)


def contexts(x, d, r, k):
    n = len(x)
    lo, hi = d + k - 1, r + k - 1
    t = np.arange(lo, n - hi, dtype=np.int64)
    cid = np.zeros(len(t), dtype=np.int64)
    for j in range(k):
        cid = cid * V + x[t - d - j].astype(np.int64)
    for j in range(k):
        cid = cid * V + x[t + r + j].astype(np.int64)
    return cid, x[t].astype(np.int64)


def cond_ce(train, test, d, r, k):
    ctr, mtr = contexts(train, d, r, k)
    cte, mte = contexts(test, d, r, k)
    ncx = V ** (2 * k)
    flat = np.bincount(ctr * V + mtr, minlength=ncx * V)
    tot = flat.reshape(ncx, V).sum(1)
    num = flat[cte * V + mte].astype(np.float64) + ALPHA
    den = tot[cte].astype(np.float64) + ALPHA * V
    return float(-np.mean(np.log(num / den)))


def marginal_ce(train, test):
    c = np.bincount(train.astype(np.int64), minlength=V)
    p = (c + ALPHA) / (c.sum() + ALPHA * V)
    return float(-np.mean(np.log(p[test.astype(np.int64)])))


def main():
    print("\nCONTEXT-WIDTH CONTROL  (exact, k=1 vs k=2)", flush=True)
    print("=" * 70, flush=True)
    x = load_corpus()
    half = len(x) // 2
    train, test = x[:half], x[half:]
    H0 = marginal_ce(train, test)
    print(f"corpus {len(x):,} symbols   marginal H(X) = {H0:.4f} nats\n",
          flush=True)

    tables = {}
    for k in (1, 2):
        I = np.zeros((MAXD + 1, MAXD + 1))
        for d in range(1, MAXD + 1):
            for r in range(1, MAXD + 1):
                I[d, r] = H0 - cond_ce(train, test, d, r, k)
            print(f"  k={k}  depth {d}/{MAXD} done", flush=True)
        tables[k] = I

    print("\n" + "=" * 70, flush=True)
    print("A.  RECOVERABLE INFORMATION AT SYMMETRIC DEPTH  i_k(d,d)")
    print("=" * 70)
    print(f"  {'depth':>6} {'k=1':>11} {'k=2':>11} {'gain':>8}")
    print("  " + "-" * 38)
    for d in range(1, MAXD + 1):
        a, b = tables[1][d, d], tables[2][d, d]
        g = b / a if abs(a) > 1e-6 else float("nan")
        print(f"  {d:>6} {a:>11.6f} {b:>11.6f} {g:>7.2f}x")

    print("\n" + "=" * 70)
    print("B.  SPAN CURVES")
    print("=" * 70)
    slopes = {}
    for k in (1, 2):
        I = tables[k]
        ls = np.arange(1, MAXD + 1)
        S = np.array([sum(I[p, l + 1 - p] for p in range(1, l + 1))
                      for l in ls])
        ib = S / ls
        slopes[k] = np.polyfit(np.log(ls[1:]),
                               np.log(np.maximum(ib[1:], 1e-9)), 1)[0]
        print(f"\n  context width k={k}")
        print(f"    {'l':>3} {'S(l)':>10} {'ibar(l)':>10} {'ibar*l':>9}")
        for j, l in enumerate(ls):
            print(f"    {l:>3} {S[j]:>10.6f} {ib[j]:>10.6f} {ib[j]*l:>9.4f}")
        print(f"    log-log slope of ibar(l) = {slopes[k]:+.4f}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    drift = abs(slopes[2] - slopes[1])
    deep = tables[2][5, 5] / max(tables[1][5, 5], 1e-9)
    print(f"  slope k=1 = {slopes[1]:+.4f}")
    print(f"  slope k=2 = {slopes[2]:+.4f}")
    print(f"  drift     = {drift:.4f}")
    print(f"  deep-position gain i_2(5,5)/i_1(5,5) = {deep:.2f}x")
    print()
    if drift < 0.25:
        print("  => EFFECT IS REAL. A 729x larger conditioning set does not")
        print("     move the decay exponent. The 1/l scaling reflects the")
        print("     language, not the estimator's Markov assumption.")
    else:
        print("  => ARTEFACT WARNING. The exponent moves with context width;")
        print("     the earlier agreement with theory was partly built in.")
    print("=" * 70)

    np.savez(str(__import__("pathlib").Path(__file__).parent / "control_results.npz"),
             I1=tables[1], I2=tables[2], H0=H0,
             s1=slopes[1], s2=slopes[2])


if __name__ == "__main__":
    main()
