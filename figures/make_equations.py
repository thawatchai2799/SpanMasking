"""make_equations.py -- render every display equation as a 600 dpi transparent
PNG, using matplotlib mathtext (no external LaTeX install needed, so this is
guaranteed to run in the build environment).

Why images and not native Word equations (OMML): the `docx` library emits
spec-valid OMML, but this environment's LibreOffice -- the tool used to
render a PDF preview for visual QA throughout this project -- silently drops
every <m:oMath> element on DOCX import. Confirmed three ways, including with
a hand-written OOXML file that does not depend on the docx library at all.
That means an OMML equation could not be visually verified before shipping,
which is the discipline that caught real errors in this manuscript before.
Images can be checked the same way the six figures were, so images are the
path that keeps every equation checkable.

Matplotlib's mathtext supports most LaTeX math syntax but has NO array/align
environment, so multi-line derivations (e.g. a chain of three equalities) are
built here as several single-line mathtext strings stacked with a shared '='
column, not as one LaTeX array.

Run:  python make_equations.py
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parent / "equations"
DPI = 600
FONT_SIZE = 12.5

plt.rcParams.update({
    "mathtext.fontset": "stix",   # a Times-like math font, close to the body serif
})


# ------------------------------------------------------------- single-line
SINGLE = {
    # ---------------------------------------------------- Section 3
    "thm1": r"$I(X_S\,;\,X_{S^c}) = I(X_S\,;\,X_{\partial S}), "
            r"\quad |\partial S| \leq 2 m_0 k$",
    "cor11": r"$I(X_S\,;\,X_{S^c}) = \dfrac{1}{2}\log\!\left["
             r"\dfrac{1 - r^{2(l+1)}}{(1-r^2)^2}\right]$",
    "cor12": r"$I(X_S\,;\,X_{S^c}) \leq H(X_{\partial S}) \leq "
             r"2 m_0 k \log|V| = O(m/l)$",
    "thm2": r"$\dfrac{d\|W\|_F^2}{dt} = -2\omega \|W\|_F^2,"
            r"\qquad \|W(t)\|_F = \|W(0)\|_F\, e^{-\omega t}$",
    "ldb_def": r"$L_{LDB} = L_{cos} + L_{ent} + \dfrac{\kappa}{2}"
               r"\left(\dfrac{\mathrm{tr}\,C}{d} - 1\right)^{2}$",
    "eta_star": r"$\lambda^\ast = \dfrac{\beta}{2\eta} - \varepsilon, "
                r"\qquad \eta^\ast = \dfrac{\beta}{2\varepsilon}$",

    # ---------------------------------------------------- Appendix A
    "a1_claim": r"$I(X_S\,;\,X_{S^c}) = I(X_S\,;\,X_{\partial S})$",
    "a1_chain": r"$I(X_S\,;\,X_{S^c}) = I(X_S\,;\,X_{\partial S}) + "
                r"I(X_S\,;\,X_R \mid X_{\partial S})$",
    "a1_bound": r"$|\partial S| \leq 2\, m_0\, k$",
    "a2_closed": r"$I(X_S\,;\,X_{S^c}) = \dfrac{1}{2}\log\!\left["
                 r"\dfrac{1 - r^{2(l+1)}}{(1-r^2)^2}\right]$",
    "a3_bound": r"$I(X_S\,;\,X_{S^c}) = I(X_S\,;\,X_{\partial S}) \leq "
                r"H(X_{\partial S}) \leq |\partial S|\log|V| "
                r"\leq 2 m_0 k \log|V| = O(m/l)$",
    "a4_flow": r"$\dfrac{dW}{dt} = -\nabla_{W} L_{cos} - \omega W$",
    "a4_solution": r"$\|W(t)\|_F = \|W(0)\|_F\, e^{-\omega t}$",
    "a5_cdelta": r"$C_\delta = I_d - (1-\delta)\,v v^{\top}, "
                 r"\qquad v = \dfrac{e_1+e_2}{\sqrt{2}}$",
    "a6_lent": r"$L_{ent}(C) = -\dfrac{\beta}{2}\,\log\det(C + \varepsilon I)$",
    "a6_ray": r"$\det(cC_0 + \varepsilon I) \geq c^{\,d}\det(C_0)"
              r" \ \Rightarrow\ L_{ent} \to -\infty \ \ \mathrm{as}\ \ c \to \infty$",
    "a6_ldb": r"$L_{LDB} = L_{cos} + L_{ent} + \dfrac{\kappa}{2}"
              r"\left(\dfrac{\mathrm{tr}\,C}{d} - 1\right)^{2}$",
    "a7_obj": r"$f(\lambda) = -\dfrac{\beta}{2}\log(\lambda+\varepsilon) + \eta\lambda$",
    "a7_threshold": r"$\eta^\ast = \dfrac{\beta}{2\varepsilon}$",
    "a7_separates": r"$\log\det(C+\varepsilon I) = \sum_i \log(\lambda_i+\varepsilon)$",
    "a1_indep": r"$X_S \perp X_R \mid X_{\partial S}$",
    "a7_lambda_star": r"$\lambda^\ast = \dfrac{\beta}{2\eta} - \varepsilon$",
}

# --------------------------------------------------------------- multi-line
# each entry: list of (left-hand side, right-hand side) pairs, aligned on
# a shared "=" column, rendered top to bottom
MULTI = {
    "a4_deriv": [
        (r"$\dfrac{d\|W\|_F^2}{dt}$", r"$= 2\langle W,\, \dot W \rangle$"),
        ("", r"$= -2\langle W,\, \nabla_{W} L_{cos} \rangle "
             r"- 2\omega \|W\|_F^2$"),
        ("", r"$= -2\omega \|W\|_F^2$"),
    ],
}


def render_single(key, latex):
    fig = plt.figure(figsize=(6, 1))
    fig.text(0.02, 0.5, latex, fontsize=FONT_SIZE, ha="left", va="center")
    OUT.mkdir(exist_ok=True)
    path = OUT / (key + ".png")
    fig.savefig(path, dpi=DPI, transparent=True,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def render_multi(key, rows):
    """Stack single-line mathtext strings, right-aligning the left half and
    left-aligning the right half on a shared column, so '=' signs line up
    the way a LaTeX align environment would show them. Laid out on a large,
    fixed-size canvas in DATA (axes-fraction-independent) coordinates so
    that bbox_inches='tight' can crop it to the actual ink afterwards --
    the earlier version pre-computed a canvas size and left generous blank
    margins because the size estimate did not match true glyph extents."""
    n = len(rows)
    fig = plt.figure(figsize=(8, 0.55 * n))
    col_x = 0.35   # right edge of the left-hand column, in figure fraction
    gap = 0.02
    for i, (lhs, rhs) in enumerate(rows):
        y = 1 - (i + 0.5) / n
        if lhs:
            fig.text(col_x, y, lhs, fontsize=FONT_SIZE, ha="right", va="center")
        if rhs:
            fig.text(col_x + gap, y, rhs, fontsize=FONT_SIZE, ha="left", va="center")
    OUT.mkdir(exist_ok=True)
    path = OUT / (key + ".png")
    fig.savefig(path, dpi=DPI, transparent=True,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("rendering %d single-line + %d multi-line equations to %s"
          % (len(SINGLE), len(MULTI), OUT))
    for k, latex in SINGLE.items():
        p = render_single(k, latex)
        print("  %-16s %6.0f KB  %s" % (k, p.stat().st_size / 1024, latex[:55]))
    for k, rows in MULTI.items():
        p = render_multi(k, rows)
        print("  %-16s %6.0f KB  (%d lines)" % (k, p.stat().st_size / 1024, len(rows)))
    print("done.")
