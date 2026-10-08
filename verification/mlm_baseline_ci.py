"""mlm_baseline_ci.py -- the pure span-MLM baseline of Section 6.4.

Added after peer review at a reviewer's request. Compares the conventional
span-masked MLM objective (runs R2_Bmlmonly_seed0-14: cross-entropy on the
masked positions only, no cosine term, no target encoder, every other
setting identical to cell B) against the cosine JEPA cell B on all four
probes. Every number in Section 6.4 and Table 4 is printed by this script
and copied verbatim.

  welch    two-sided 95% Welch t-interval on the difference of means
           (Welch-Satterthwaite degrees of freedom); printed in Table 4
  paired   two-sided 95% t-interval on per-seed differences (df = 14),
           a sensitivity check; the two constructions agree on every verdict
  gate     the pre-registered two-point margin over the random-initialised
           encoder on both semantic probes

Run from the repository root:  python verification/mlm_baseline_ci.py
"""
import json
import pathlib

import numpy as np
from scipy import stats

R = pathlib.Path(__file__).resolve().parent.parent / "results"
if not R.exists():
    R = pathlib.Path(__file__).resolve().parent.parent / "paper" / "results_final"

def load(rid):
    return json.loads((R / (rid + ".json")).read_text())

def cell(prefix, n, key):
    return np.array([load(f"{prefix}_seed{s}")["probes"][key] for s in range(n)])

def welch_ci(a, b):
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(0.975, df)
    return d, d - t * se, d + t * se, df

def paired_ci(a, b):
    diff = a - b
    t = stats.t.ppf(0.975, len(diff) - 1)
    h = t * diff.std(ddof=1) / np.sqrt(len(diff))
    return diff.mean(), diff.mean() - h, diff.mean() + h

if __name__ == "__main__":
    P = 100.0
    base = load("BASE_randominit")["probes"]
    print("pure span-MLM (R2_Bmlmonly, n=15) against cosine JEPA cell B (n=15)")
    for key, label in [("agnews", "AG News"), ("sst2", "SST-2"),
                       ("wordcontent", "word content"), ("pos", "POS")]:
        m = cell("R2_Bmlmonly", 15, key) * P
        b = cell("P1_B", 15, key) * P
        d, lo, hi, df = welch_ci(m, b)
        pm, plo, phi = paired_ci(m, b)
        if hi < 0:
            verdict = "MLM below cosine"
        elif lo > 0:
            verdict = "MLM above cosine"
        else:
            verdict = "no resolved difference"
        print(f"  {label:13s} MLM {m.mean()/P:.4f}  cosine {b.mean()/P:.4f}  "
              f"diff {d:+.2f}  welch [{lo:+.2f}, {hi:+.2f}] (df={df:.1f})  "
              f"paired {pm:+.2f} [{plo:+.2f}, {phi:+.2f}]  -> {verdict}")
    ag = cell("R2_Bmlmonly", 15, "agnews").mean() * P - base["agnews"] * P
    ss = cell("R2_Bmlmonly", 15, "sst2").mean() * P - base["sst2"] * P
    print(f"  gate: margin over random-init AG News {ag:+.2f}, SST-2 {ss:+.2f} -> "
          f"{'pass' if ag >= 2.0 and ss >= 2.0 else 'FAIL'}")
    er = np.mean([load(f"R2_Bmlmonly_seed{s}")["diagnostics"][-1]["eff_rank"]
                  for s in range(15)])
    print(f"  final effective rank of masked-position encoder states: {er:.1f} "
          f"(not comparable to the predictor-output ranks of Table 3)")
