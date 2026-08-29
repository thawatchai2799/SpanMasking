"""table3_ci.py -- the confidence intervals of Table 3, from the run records.

Every interval in Table 3 of the paper is produced by this script and
copied verbatim; nothing is typed from memory. Two constructions are
reported:

  welch    two-sided 95% Welch t-interval on the difference of means
           (arm variance over its own seeds, baseline variance over its
           fifteen, Welch-Satterthwaite degrees of freedom). This is the
           interval printed in Table 3.

  paired   two-sided 95% t-interval on the per-seed differences over the
           nine seeds the arm and the baseline share (df = 8), reported
           as a sensitivity check. The two constructions agree on every
           verdict.

The pre-registered no-benefit criterion: the interval must exclude a
one-point gain (upper bound < +1.0 accuracy points).

Run from the repository root:  python verification/table3_ci.py
"""
import json
import pathlib

import numpy as np
from scipy import stats

R = pathlib.Path(__file__).resolve().parent.parent / "results"
if not R.exists():
    R = pathlib.Path(__file__).resolve().parent.parent / "paper" / "results_final"

def probes(rid, key):
    return json.loads((R / (rid + ".json")).read_text())["probes"][key]

def cell(prefix, n, key):
    return np.array([probes(f"{prefix}_seed{s}", key) for s in range(n)])

def welch_ci(arm, base):
    d = arm.mean() - base.mean()
    va, vb = arm.var(ddof=1) / len(arm), base.var(ddof=1) / len(base)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(arm) - 1) + vb ** 2 / (len(base) - 1))
    t = stats.t.ppf(0.975, df)
    return d, d - t * se, d + t * se, df

def paired_ci(arm, base):
    n = min(len(arm), len(base))
    diff = arm[:n] - base[:n]
    m, sd = diff.mean(), diff.std(ddof=1)
    t = stats.t.ppf(0.975, n - 1)
    half = t * sd / np.sqrt(n)
    return m, m - half, m + half

if __name__ == "__main__":
    P = 100.0
    for key, label in [("agnews", "AG News"), ("sst2", "SST-2")]:
        base = cell("P1_B", 15, key) * P
        print(f"{label}: baseline mean {base.mean()/P:.4f} (n=15)")
        for pre, name in [("S6_Bvicreg", "B+VICReg"), ("S6_Bldb", "B+LDB")]:
            arm = cell(pre, 9, key) * P
            d, lo, hi, df = welch_ci(arm, base)
            pm, plo, phi = paired_ci(arm, base)
            verdict = "no benefit" if hi < 1.0 else "inconclusive"
            print(f"  {name:9s} mean {arm.mean()/P:.4f}  vs base {d:+.2f}"
                  f"  welch [{lo:+.2f}, {hi:+.2f}] (df={df:.1f}) -> {verdict}"
                  f"  | paired {pm:+.2f} [{plo:+.2f}, {phi:+.2f}]")
