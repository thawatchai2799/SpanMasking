"""
IS THEOREM 4's BOUND VACUOUS FOR ANISOTROPIC COVARIANCE?
========================================================
Remark 4.1 established that the bound

    lam_min >= exp(-2c/beta) * ((d-1)/(tr C + d*eps))^(d-1) - eps

is tight to a factor of e for ISOTROPIC C. Real embedding covariances are
never isotropic; they have power-law spectra. The AM-GM step in the proof is
tight only when the eigenvalues are equal, so the bound should degrade as
(GM/AM)^(d-1) -- potentially to nothing at d = 192.

This decides whether Section 3.5 can keep its quantitative claim.
"""

import numpy as np

EPS = 1e-4
BETA = 1.0


def spectrum(kind, d):
    j = np.arange(1, d + 1, dtype=float)
    if kind == "isotropic":
        w = np.ones(d)
    elif kind == "mild power law":
        w = j ** -0.5
    elif kind == "power law 1/j":
        w = j ** -1.0
    elif kind == "steep 1/j^1.5":
        w = j ** -1.5
    elif kind == "exponential":
        w = np.exp(-0.05 * j)
    elif kind == "one small eig":
        w = np.ones(d); w[-1] = 0.05
    return w / w.sum() * d          # normalise tr(C) = d


def bound(w, d):
    c = -(BETA / 2.0) * np.sum(np.log(w + EPS))
    return np.exp(-2 * c / BETA) * ((d - 1) / (w.sum() + d * EPS)) ** (d - 1) - EPS


def main():
    print("\nANISOTROPY STRESS TEST FOR THEOREM 4")
    print("=" * 78)
    print("  tr(C) = d in every case, so the scale-covariance argument of")
    print("  Remark 4.1 applies identically. Only the SHAPE differs.\n")

    kinds = ["isotropic", "mild power law", "power law 1/j",
             "steep 1/j^1.5", "exponential", "one small eig"]

    for d in [16, 64, 192]:
        print(f"  d = {d}")
        print(f"    {'spectrum':>16} {'true lam_min':>14} {'bound':>13}"
              f" {'bound/true':>12} {'status':>10}")
        print("    " + "-" * 68)
        for kind in kinds:
            w = spectrum(kind, d)
            lm = w.min()
            b = bound(w, d)
            ratio = b / lm if lm > 0 else float("nan")
            if b <= 0:
                status = "VACUOUS"
            elif ratio > 1e-3:
                status = "usable"
            else:
                status = "useless"
            print(f"    {kind:>16} {lm:>14.3e} {b:>13.3e}"
                  f" {ratio:>12.2e} {status:>10}")
        print()

    print("=" * 78)
    print("DIAGNOSIS: how loose is the AM-GM step?")
    print("=" * 78)
    print(f"  {'spectrum':>16} " + "".join(f"{'d='+str(d):>14}" for d in [16, 64, 192]))
    print("  " + "-" * 60)
    for kind in kinds:
        cells = []
        for d in [16, 64, 192]:
            w = spectrum(kind, d)
            tail = np.sort(w)[1:] + EPS          # all but the smallest
            gm = np.exp(np.mean(np.log(tail)))
            am = tail.mean()
            cells.append((gm / am) ** (d - 1))
        print(f"  {kind:>16} " + "".join(f"{c:>14.2e}" for c in cells))
    print("\n  The last column is exactly the factor by which the bound")
    print("  under-states lam_min. Anything below ~1e-3 makes the")
    print("  quantitative claim of Theorem 4 unusable at that width.")
    print("=" * 78)


if __name__ == "__main__":
    main()
