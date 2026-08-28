"""Does the first-half/second-half split confound the bridge estimates?"""
import numpy as np
src = open("bridge_control2.py").read().split("def main()")[0]
g = {}; exec(src, g)

x = g["load_corpus"]()
n = len(x)

# split A: contiguous halves (what we used) -- different authors each side
trA, teA = x[:n//2], x[n//2:]

# split B: interleaved 100k-symbol blocks -- same author mix both sides
blocks = [x[i:i+100_000] for i in range(0, n, 100_000)]
trB = np.concatenate(blocks[0::2]); teB = np.concatenate(blocks[1::2])

for name, (tr, te) in [("contiguous halves", (trA, teA)),
                       ("interleaved blocks", (trB, teB))]:
    H0 = g["marginal_ce"](tr, te)
    print(f"\n{name}:  marginal H = {H0:.4f} nats   "
          f"(train {len(tr):,} / test {len(te):,})")
    print(f"  {'depth':>6} {'i_k=1':>10} {'i_k=2':>10}")
    vals = {}
    for d in (1, 2, 3, 4, 5):
        a = H0 - g["cond_ce"](tr, te, d, d, 1)
        b = H0 - g["cond_ce"](tr, te, d, d, 2)
        vals[d] = (a, b)
        print(f"  {d:>6} {a:>10.6f} {b:>10.6f}")
    for k, idx in [("k=1", 0), ("k=2", 1)]:
        d = np.array([1, 2, 3]); y = np.log([vals[i][idx] for i in d])
        lam = -np.polyfit(d, y, 1)[0]
        print(f"    {k}: lambda = {lam:.4f}")
    print(f"    ratio i(3,3)/i(1,1) at k=2: {vals[3][1]/vals[1][1]:.4f}")
