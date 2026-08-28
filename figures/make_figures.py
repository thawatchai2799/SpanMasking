"""
make_figures.py -- figures 1 to 4 of the span-masking paper.

Every number plotted here is produced in this file or copied from the
verification scripts named in each function's docstring; nothing is typed
from memory. Figures 5 and 6 need results/summary.csv and are added once
the training ledger has finished.

Figure numbers follow the order of first citation in the manuscript
(Section 3.1, 3.1, 3.3, 3.5); the file names carry the content so that
renumbering stays visible.

Output: figures/figN_name.{pdf,eps,png}
  pdf, eps  vector, resolution independent
  png       raster at 600 dpi, for drafts and slides

Run:  python make_figures.py
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent / "figures"
DPI = 600
# Output formats. The log line is derived from this list rather than
# written out by hand: an earlier version printed "pdf,eps,png" after the
# code had stopped emitting eps, and stale eps files on disk made the
# claim look true.
FORMATS = ("pdf", "png")

# ----------------------------------------------------------------- style
# Colour, but not colour alone. The palette is Okabe-Ito, which stays
# distinguishable under every common form of colour blindness, and every
# series is ALSO separated by line style and marker so the figures survive
# a greyscale printout. MAKE is online-first and charges nothing for
# colour, so there is no reason to publish these in grey.
BLUE    = "#0072B2"
VERM    = "#D55E00"
GREEN   = "#009E73"
ORANGE  = "#E69F00"
PURPLE  = "#CC79A7"
SKY     = "#56B4E9"
YELLOW  = "#F0E442"
INK     = "#1A1A1A"
GRID    = "#D9D9D9"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#4D4D4D",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.7,
    "lines.markersize": 4.5,
    "figure.constrained_layout.use": True,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,      # embed TrueType, not Type 3: required by
    "ps.fonttype": 42,       # many publishers' preflight checks
})

# (colour, linestyle, marker) triples: never colour alone
SERIES = [(BLUE, "-", "o"), (VERM, "--", "s"), (GREEN, ":", "^"),
          (PURPLE, "-.", "D"), (ORANGE, (0, (3, 1, 1, 1)), "v")]

COL_W = 3.35     # inches, one column
DBL_W = 6.90     # inches, full width


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in FORMATS:
        fig.savefig(OUT / ("%s.%s" % (name, ext)), dpi=DPI)
    plt.close(fig)
    print("  wrote %s.{%s}" % (name, ",".join(FORMATS)))


# ------------------------------------------------------- theory formulas
def mi_block_ar1(l, rho):
    """I(X_S ; X_S^c) for one interior block of length l in a Gaussian
    AR(1) chain. Closed form of Corollary 1.1; agrees with direct
    covariance computation to 1e-13 nats (verify_thm1.py, table T2/T3)."""
    return 0.5 * np.log((1.0 - rho ** (2 * (l + 1))) / (1.0 - rho ** 2) ** 2)


# ------------------------------------------------------------- figure 1
def fig1_schematic():
    """Conceptual figure: the context speaks to the masked set only
    through the block boundaries, so the number of channels grows with
    the block count k and not with the masked-token count m."""
    fig, axes = plt.subplots(2, 1, figsize=(DBL_W, 2.5))
    T = 32

    configs = [
        ("(a) k = 4 blocks of length l = 2   (m = 8 masked tokens)",
         [(4, 2), (11, 2), (18, 2), (25, 2)]),
        ("(b) k = 1 block of length l = 8   (m = 8 masked tokens)",
         [(12, 8)]),
    ]

    for ax, (title, blocks) in zip(axes, configs):
        masked = np.zeros(T, dtype=bool)
        for s, l in blocks:
            masked[s:s + l] = True
        boundary = np.zeros(T, dtype=bool)
        for s, l in blocks:
            if s - 1 >= 0:
                boundary[s - 1] = True
            if s + l < T:
                boundary[s + l] = True

        for i in range(T):
            if masked[i]:
                fc, ec, hatch, lw = SKY, BLUE, "///", 0.8
            elif boundary[i]:
                fc, ec, hatch, lw = YELLOW, VERM, None, 1.8
            else:
                fc, ec, hatch, lw = "#FFFFFF", "#C4C4C4", None, 0.6
            ax.add_patch(plt.Rectangle(
                (i, 0), 0.86, 1, facecolor=fc, edgecolor=ec,
                linewidth=lw, hatch=hatch))

        n_b = int(boundary.sum())
        ax.text(T + 0.6, 0.5, "%d boundary\npositions" % n_b,
                va="center", ha="left", fontsize=8.5, color=VERM,
                fontweight="bold")
        ax.set_xlim(-0.5, T + 4.5)
        ax.set_ylim(-0.15, 1.15)
        ax.set_title(title, loc="left", pad=3, color=INK)
        ax.axis("off")

    fig.text(0.012, 0.5, "", rotation=90)
    save(fig, "fig1_boundary_schematic")


# ------------------------------------------------------------- figure 2
def fig2_theorem1():
    """Theorem 1 and Corollary 1.1. Left: total information saturates in
    the span length at -log(1-rho^2). Right: at a fixed masking budget
    the information is proportional to the block count. Curves are the
    closed form verified in verify_thm1.py (max error 1.5e-13 nats)."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DBL_W, 2.7))

    ls = np.arange(1, 31)
    # curves are labelled at their right-hand end rather than in a legend:
    # a legend box here overlapped both the curves and the asymptote lines
    for j, rho in enumerate([0.60, 0.75, 0.85, 0.95]):
        c, ls_, mk = SERIES[j]
        y = mi_block_ar1(ls, rho)
        axL.plot(ls, y, linestyle=ls_, marker=mk, color=c, markevery=(j, 4))
        axL.axhline(-np.log(1 - rho ** 2), color=c,
                    linewidth=0.7, linestyle=(0, (1, 3)), alpha=0.7)
        axL.annotate(r"$\rho = %.2f$" % rho, xy=(30.4, y[-1]),
                     va="center", ha="left", fontsize=8, color=c,
                     fontweight="bold", annotation_clip=False)
    axL.set_xlim(0, 30)
    axL.set_ylim(0, 2.62)
    axL.set_xlabel(r"span length $l$")
    axL.set_ylabel(r"$I(X_S; X_{S^c})$  (nats)")
    axL.set_title(r"(a) one block: saturation in $l$", loc="left")
    axL.text(0.5, 2.50, r"dotted lines: $-\log(1-\rho^2)$", fontsize=7.5,
             color="#666666")

    # fixed budget m = 32 masked tokens, arranged as k blocks of length m/k
    m, rho = 32, 0.85
    ks = np.array([1, 2, 4, 8, 16, 32])
    per_block = mi_block_ar1(m / ks, rho)
    axR.plot(ks, ks * per_block, linestyle="-", marker="o", color=BLUE,
             label="total  " + r"$I \propto k$")
    axR.plot(ks, per_block, linestyle="--", marker="s", color=VERM,
             label="per block (bounded)")
    axR.plot(ks, ks * per_block / m, linestyle=":", marker="^", color=GREEN,
             label=r"per masked token  $\propto 1/l$")
    axR.set_xscale("log", base=2)
    axR.set_yscale("log")
    axR.set_xlabel(r"block count $k$   (fixed budget $m = 32$, $l = m/k$)")
    axR.set_ylabel("information (nats)")
    axR.set_title(r"(b) fixed budget: evidence scales with $k$", loc="left")
    axR.legend(frameon=False, loc="upper left")

    save(fig, "fig2_theorem1_saturation")


# ------------------------------------------------------------- figure 3
def fig4_english():
    """English text, from bridge_experiment.py section B (character-level
    entropy-rate estimates on a held-out corpus). Numbers copied from that
    script's output table; the estimator caveats are stated in Section
    3.5 and repeated in the caption."""
    # read the arrays the verification script saved, not the subset it
    # prints: the printed table omits rows, and fitting the subset gave
    # -1.055 where the script reports -1.035
    d = np.load(pathlib.Path(__file__).resolve().parent
                / "bridge_results.npz")
    l, S, ibar = d["ls"].astype(float), d["S"], d["ibar"]
    fitted = float(d["slope"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DBL_W, 2.6))

    # bridge_experiment.py fits the decay over l >= 4 only, because S(l)
    # has not saturated below that; the figure marks the same split
    FIT0 = 3          # index of l = 4, matching ls[3:] in that script

    axL.axvspan(0.5, 4, color=ORANGE, alpha=0.13, zorder=0)
    axL.plot(l, S, linestyle="-", marker="o", color=BLUE,
             label="measured  " + r"$S(l)$")
    axL.axhline(S[-1], color=VERM, linewidth=1.0, linestyle=(0, (1, 3)))
    axL.set_xlabel(r"span length $l$ (characters)")
    axL.set_ylabel("recoverable information (nats)")
    axL.set_title("(a) total per span saturates", loc="left")
    axL.text(2.2, S.min() + 0.01, "pre-\nsaturation", fontsize=7.5,
             color="#A05000", ha="center", va="bottom", fontweight="bold")
    axL.legend(frameon=False, loc="lower right")

    axR.axvspan(0.5, 4, color=ORANGE, alpha=0.13, zorder=0)
    axR.loglog(l, ibar, linestyle="-", marker="o", color=BLUE,
               label="measured per token")
    lf = l[FIT0:]
    ref = ibar[FIT0] * (lf / lf[0]) ** (-1.0)
    axR.loglog(lf, ref, linestyle="--", color=VERM, linewidth=1.4,
               label=r"slope $-1$, anchored at $l=4$")
    axR.set_xlabel(r"span length $l$ (characters)")
    axR.set_ylabel(r"$\bar{\imath}(l)$  (nats per masked token)")
    axR.set_title(r"(b) decay over the fit range: $%+.2f$ vs $-1$"
                  % fitted, loc="left")
    axR.legend(frameon=False, loc="lower left")

    save(fig, "fig4_english_decay")


# ------------------------------------------------------------- figure 4
def fig3_eps_law():
    """Propositions 5-6: the log-determinant objective's collapse
    threshold is eta* = beta/(2 eps). All numbers are the canonical
    sweep of canonical_thresholds.py (d=24, n=384, 12000 steps,
    criterion lam_min < 1e-6), which is the single source for every
    threshold quoted in the paper. The bracket is [last alive, first
    dead]; no interpolated centre point is drawn, because none was
    measured."""
    # canonical_thresholds.py, table rows LDB and VICReg
    eps = np.array([1e-2, 1e-3, 1e-4])
    predicted = np.array([50.0, 500.0, 5000.0])
    alive = np.array([52.0, 505.0, 4894.0])
    dead = np.array([53.0, 517.0, 5006.0])
    vic_nu = np.array([1.0, 16.0, 256.0, 4096.0])
    vic_alive = np.array([7.0, 32.0, 329.0, 1389.0])
    vic_dead = np.array([7.0, 33.0, 337.0, 1421.0])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DBL_W, 2.7),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    grid = np.logspace(-4.35, -1.65, 60)
    axL.loglog(grid, 1.0 / (2 * grid), linestyle="-", color=BLUE,
               linewidth=1.6, zorder=2,
               label=r"theory  $\eta^\ast = \beta/(2\varepsilon)$")
    axL.plot(eps, alive, linestyle="none", marker="o", color=VERM,
             markersize=7, markeredgecolor="white", markeredgewidth=0.8,
             zorder=3, label="measured threshold")
    axL.axhspan(vic_alive.min(), vic_alive.max(), color=GREEN, alpha=0.13,
                zorder=0)
    axL.text(3.4e-2, vic_alive.min() * 1.15,
             "VICReg, " + r"$\nu = 1 \dots 4096$",
             fontsize=7.5, color=GREEN, va="bottom", ha="right",
             fontweight="bold")
    axL.set_xlabel(r"ridge $\varepsilon$")
    axL.set_ylabel(r"collapse pressure $\eta^\ast$")
    axL.set_title(r"(a) the $\varepsilon$ law over three decades",
                  loc="left")
    axL.set_ylim(3, 3e4)
    axL.legend(frameon=False, loc="upper right", handlelength=1.6,
               borderpad=0.2)
    axL.grid(True, which="major", linewidth=0.4, color=GRID, zorder=0)

    # the measured brackets are narrower than a marker on panel (a), so
    # the agreement is shown as a ratio instead of being hidden
    mid = 0.5 * (alive + dead)
    axR.axhspan(0.95, 1.05, color=BLUE, alpha=0.10, zorder=0)
    axR.semilogx(eps, mid / predicted, linestyle="none", marker="o",
                 color=VERM, markersize=7, markeredgecolor="white",
                 markeredgewidth=0.8, zorder=3)
    for i, e in enumerate(eps):
        axR.plot([e, e], [alive[i] / predicted[i], dead[i] / predicted[i]],
                 color=VERM, linewidth=1.6, zorder=2)
    axR.axhline(1.0, color=BLUE, linewidth=1.2, linestyle="--", zorder=1)
    axR.set_xlabel(r"ridge $\varepsilon$")
    axR.set_ylabel(r"measured $/$ predicted")
    axR.set_title(r"(b) agreement within $5\%$", loc="left")
    axR.set_ylim(0.90, 1.10)
    axR.grid(True, which="major", linewidth=0.4, color=GRID)
    axR.text(1.2e-4, 0.915,
             "bars: [last alive, first dead]\nfitted exponent $+0.986$",
             fontsize=7.5, va="bottom")

    save(fig, "fig3_eps_threshold_law")


# ------------------------------------------------------------- figure 5
def fig5_p1b_curve():
    """The span-length curve at fixed masking ratio, and the effect sizes
    the same apparatus did resolve. Every value is read from the run
    records in results_final/; nothing is transcribed."""
    import json

    R = pathlib.Path(__file__).resolve().parent / "results_final"

    def probes(rid):
        return json.loads((R / (rid + ".json")).read_text())["probes"]

    def cell(prefix, key, n=5):
        return np.array([probes("%s_seed%d" % (prefix, s))[key]
                         for s in range(n)])

    base = probes("BASE_randominit")
    curve = [(1, "P1b_l1"), (2, "P1b_l2"), (4, "P1_B"),
             (8, "P1_D"), (16, "P1b_l16")]
    ls = np.array([c[0] for c in curve], dtype=float)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DBL_W, 2.8),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) the curve, per-seed points and cell means
    M = np.array([cell(p, "agnews") for _, p in curve]) * 100
    for j in range(M.shape[1]):
        axL.plot(ls, M[:, j], linestyle="none", marker="o", color=SKY,
                 markersize=4, zorder=2, alpha=0.85,
                 label="individual seeds" if j == 0 else None)
    axL.plot(ls, M.mean(axis=1), linestyle="-", marker="s", color=BLUE,
             markersize=6, markeredgecolor="white", markeredgewidth=0.8,
             zorder=3, label="cell mean")
    axL.axhline(base["agnews"] * 100, color=VERM, linewidth=1.3,
                linestyle="--", zorder=1)
    axL.text(1.05, base["agnews"] * 100 + 0.10, "random-init encoder",
             fontsize=7.5, va="bottom", color=VERM, fontweight="bold")
    axL.axhspan(base["agnews"] * 100, base["agnews"] * 100 + 2.0,
                color=ORANGE, alpha=0.13, zorder=0)
    axL.text(16, base["agnews"] * 100 + 1.0,
             "below the\npre-registered\n2-point gate",
             fontsize=7.5, color="#A05000", ha="right", va="center",
             fontweight="bold")
    axL.set_xscale("log", base=2)
    axL.set_xticks(ls)
    axL.set_xticklabels([str(int(x)) for x in ls])
    axL.set_xlabel(r"span length $l$   (ratio fixed at $\rho = 0.25$)")
    axL.set_ylabel("AG News probe accuracy (%)")
    axL.set_title("(a) sixteen-fold range of span length", loc="left")
    axL.legend(frameon=False, loc="lower right", fontsize=7.5)

    # (b) what the apparatus did and did not resolve
    labels, vals, kinds = [], [], []
    for l, p in curve:
        if p == "P1_B":
            continue
        d = (cell(p, "agnews") - cell("P1_B", "agnews")) * 100
        labels.append(r"$l=%d$ vs $l=4$" % l)
        vals.append(d.mean())
        kinds.append("geometry")
    for name, pre, n in (("+ MLM", "P4_Bmlm", 5),
                         ("+ VICReg", "S6_Bvicreg", 3),
                         ("+ LDB", "S6_Bldb", 3)):
        d = cell(pre, "agnews", n).mean() - cell("P1_B", "agnews").mean()
        labels.append(name)
        vals.append(d * 100)
        kinds.append("objective")
    d = (cell("P1_Aeq", "agnews").mean() - cell("P1_A", "agnews").mean()) * 100
    labels.append("2x steps")
    vals.append(d)
    kinds.append("objective")

    y = np.arange(len(vals))[::-1]
    for yi, v, k in zip(y, vals, kinds):
        axR.barh(yi, v, height=0.62,
                 color=SKY if k == "geometry" else (VERM if v < 0 else GREEN),
                 edgecolor="white", linewidth=0.8, zorder=2)
    axR.axvline(0, color=INK, linewidth=0.9, zorder=3)
    mde = 0.54
    axR.axvspan(-mde, mde, color=ORANGE, alpha=0.16, zorder=0)
    axR.set_yticks(y)
    axR.set_yticklabels(labels, fontsize=7.5)
    axR.set_xlabel("change in AG News accuracy (points)")
    axR.set_title("(b) resolved and unresolved effects", loc="left")
    axR.text(0.75, len(vals) - 0.7,
             "shaded: below the\nminimum detectable\neffect (0.54 pts)",
             fontsize=7.5, color="#A05000", va="top", fontweight="bold")

    save(fig, "fig5_span_length_curve")


# ------------------------------------------------------------- figure 6
def fig6_spectral_trajectories():
    """What the anti-collapse terms do to the spectrum, and what that buys.
    Trajectories are means over seeds, read from the diagnostics logged
    every 250 steps in each run record."""
    import json

    R = pathlib.Path(__file__).resolve().parent / "results_final"

    def diag(prefix, n, key):
        runs = [json.loads((R / ("%s_seed%d.json" % (prefix, s))).read_text())
                ["diagnostics"] for s in range(n)]
        steps = np.array([d["step"] for d in runs[0]])
        vals = np.array([[d[key] for d in r] for r in runs])
        return steps, vals.mean(axis=0)

    def probes(rid):
        return json.loads((R / (rid + ".json")).read_text())["probes"]

    arms = [("cosine only", "P1_B", 5, BLUE, "-", "o"),
            ("+ VICReg", "S6_Bvicreg", 3, VERM, "--", "s"),
            ("+ log-det", "S6_Bldb", 3, GREEN, ":", "^")]

    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(DBL_W, 2.6),
                                        gridspec_kw={"width_ratios":
                                                     [1, 1, 0.85]})

    for name, pre, n, c, lsty, mk in arms:
        s, v = diag(pre, n, "lam_min")
        axL.semilogy(s, np.maximum(v, 1e-12), linestyle=lsty, marker=mk,
                     color=c, markevery=6, label=name)
    axL.set_xlabel("training step")
    axL.set_ylabel(r"smallest eigenvalue $\lambda_{\min}$")
    axL.set_title("(a) collapse is prevented", loc="left")
    axL.legend(frameon=False, loc="center right", fontsize=7.5)
    axL.grid(True, which="major", linewidth=0.4, color=GRID)

    for name, pre, n, c, lsty, mk in arms:
        s, v = diag(pre, n, "eff_rank")
        axM.plot(s, v, linestyle=lsty, marker=mk, color=c, markevery=6)
    axM.set_xlabel("training step")
    axM.set_ylabel("effective rank")
    axM.set_title("(b) rank rises accordingly", loc="left")
    axM.grid(True, which="major", linewidth=0.4, color=GRID)

    # and the probe moves the other way
    B = np.mean([probes("P1_B_seed%d" % s)["agnews"] for s in range(5)])
    xs, ys, cs = [], [], []
    for name, pre, n, c, lsty, mk in arms:
        _, er = diag(pre, n, "eff_rank")
        acc = np.mean([probes("%s_seed%d" % (pre, s))["agnews"]
                       for s in range(n)])
        xs.append(er[-1])
        ys.append(acc * 100)
        cs.append(c)
    # each label is placed by hand: centring them all put "cosine only" on
    # top of the y-axis and ran the other two labels into each other
    offs = [(9, -3, "left"), (-4, -15, "right"), (4, 11, "left")]
    axR.plot(xs, ys, linestyle="--", color="#999999", linewidth=1.0, zorder=1)
    for x, y, c, (name, *_rest), (dx, dy, ha) in zip(xs, ys, cs, arms, offs):
        axR.plot(x, y, marker="o", markersize=9, color=c,
                 markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        axR.annotate(name, xy=(x, y), xytext=(dx, dy),
                     textcoords="offset points", ha=ha, fontsize=7.5,
                     color=c, fontweight="bold", zorder=4)
    axR.set_xlabel("final effective rank")
    axR.set_ylabel("AG News accuracy (%)")
    axR.set_title("(c) but the probe falls", loc="left")
    axR.set_xlim(min(xs) - 35, max(xs) + 35)
    axR.set_ylim(min(ys) - 1.4, max(ys) + 1.2)
    axR.grid(True, which="major", linewidth=0.4, color=GRID)

    save(fig, "fig6_spectral_trajectories")


if __name__ == "__main__":
    print("writing figures to", OUT)
    fig1_schematic()
    fig2_theorem1()
    fig3_eps_law()          # cited in Section 3.3
    fig4_english()          # cited in Section 3.5
    fig5_p1b_curve()        # cited in Section 5
    fig6_spectral_trajectories()   # cited in Section 6
    print("done. all six figures written.")
