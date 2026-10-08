"""verify_p1_ci.py -- paired confidence interval for the P1 matched-k vs
matched-ratio gap, added in response to a review asking for an effect-size
and CI on top of the pre-registered win-count/p-value.

For each seed, computes:
  gap_rho = |B - D|          (matched-ratio gap, shared reference)
  gap_k   = |Aeq - B|        (matched-k gap, gate-valid cell)
  diff    = gap_k - gap_rho  (positive = wrong direction for P1)
and reports the mean paired difference with a t-interval and a
distribution-free bootstrap interval, for both the gate-valid (Aeq-B) and
blocked (A-B) comparisons.

Run:  python verify_p1_ci.py
"""
import json
import pathlib

import numpy as np
from scipy import stats

R = pathlib.Path(__file__).resolve().parent.parent / "results"
if not R.exists():
    R = pathlib.Path(__file__).resolve().parent.parent / "paper" / "results_final"


def probes(run_id):
    return json.loads((R / (run_id + ".json")).read_text())["probes"]


def cell(prefix, key, seeds=range(5)):
    return np.array([probes("%s_seed%d" % (prefix, s))[key] for s in seeds])


def summarize(name, diff, rng=None):
    m = diff.mean()
    sd = diff.std(ddof=1)
    n = len(diff)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    t_ci = (m - tcrit * se, m + tcrit * se)

    # a fresh generator per comparison keeps every interval reproducible
    # on its own, independent of the order the comparisons are printed in
    rng = np.random.default_rng(0)
    boot_means = np.array([rng.choice(diff, size=n, replace=True).mean()
                           for _ in range(20000)])
    boot_ci = np.percentile(boot_means, [2.5, 97.5])

    print("%-24s mean %+.3f pts  sd %.2f  t-CI [%+.2f, %+.2f]  "
          "bootstrap-CI [%+.2f, %+.2f]"
          % (name, m, sd, t_ci[0], t_ci[1], boot_ci[0], boot_ci[1]))
    return m, t_ci, boot_ci


def main():
    print("Positive = matched-k gap LARGER than matched-ratio gap")
    print("(the wrong direction for P1; P1 predicts this should be negative)\n")
    rng = np.random.default_rng(0)
    # Three levels, reported in this order in Section 5.2 of the paper: the
    # registered five seeds; the ten extension seeds on their own (a
    # genuinely new sample, fixed as a ledger after the registered outcome
    # was known); and the pooled fifteen, which is secondary evidence.
    levels = ((range(5), "registered five seeds (0-4)"),
              (range(5, 15), "power extension only, ten new seeds (5-14)"),
              (range(15), "pooled fifteen seeds (secondary analysis)"))
    for seeds, label in levels:
        n = len(seeds)
        B = cell("P1_B", "agnews", seeds)
        D = cell("P1_D", "agnews", seeds)
        Aeq = cell("P1_Aeq", "agnews", seeds)
        A = cell("P1_A", "agnews", seeds)

        gap_rho = np.abs(B - D) * 100
        gap_k_eq = np.abs(Aeq - B) * 100
        gap_k_a = np.abs(A - B) * 100

        diff_eq = gap_k_eq - gap_rho
        diff_a = gap_k_a - gap_rho

        print("--- %s ---" % label)
        summarize("Aeq-B vs B-D (gate-valid)", diff_eq, rng)
        pred = int((diff_eq < 0).sum())
        w = stats.wilcoxon(diff_eq, alternative="less", method="exact")
        print("%-24s seeds in predicted direction: %d/%d, "
              "one-sided exact Wilcoxon p = %.6g" % ("", pred, n, w.pvalue))
        summarize("A-B vs B-D (blocked)", diff_a, rng)
        pred_a = int((diff_a < 0).sum())
        w_a = stats.wilcoxon(diff_a, alternative="less", method="exact")
        print("%-24s seeds in predicted direction: %d/%d, "
              "one-sided exact Wilcoxon p = %.6g\n" % ("", pred_a, n, w_a.pvalue))


if __name__ == "__main__":
    main()
