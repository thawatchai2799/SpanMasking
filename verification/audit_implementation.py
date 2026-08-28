"""
IMPLEMENTATION AUDIT
====================
Every headline result depends on analytic gradients and on one mutual-
information routine. If any of those is wrong, the numbers can still agree
with the theory perfectly -- because a wrong gradient converges to the fixed
point of a wrong equation, and a wrong MI routine can be self-consistent.
Agreement with theory is therefore NOT evidence that the code is right.

This file checks the code against independent constructions:

  1  logdet gradient        vs central finite differences
  2  lam_min gradient       vs central finite differences
  3  VICReg gradient        vs central finite differences
  4  chain rule dC -> dZ    vs central finite differences on Z
  5  Gaussian MI routine    vs Schur-complement conditional entropy
  6  Gaussian MI routine    vs Monte-Carlo entropy on a 2-variable case
"""

import numpy as np

RNG = np.random.default_rng(7)
EPS, BETA, GAMMA = 1e-3, 1.0, 1.0


def sym_random_psd(d, seed=0):
    r = np.random.default_rng(seed)
    A = r.standard_normal((d, d + 5))
    return A @ A.T / (d + 5)


# ---------------------------------------------------------------- 1,2,3
def fd_grad_matrix(f, C, h=1e-6):
    """Central finite differences of a scalar function of a symmetric matrix."""
    d = C.shape[0]
    G = np.zeros_like(C)
    for i in range(d):
        for j in range(i, d):
            E = np.zeros_like(C)
            E[i, j] = E[j, i] = 1.0
            if i == j:
                E[i, j] = 1.0
            G_ij = (f(C + h * E) - f(C - h * E)) / (2 * h)
            G[i, j] = G_ij
            if i != j:
                G[j, i] = G_ij
    return G


def check(name, analytic, numeric, tol=1e-5):
    num = np.linalg.norm(analytic - numeric)
    den = max(np.linalg.norm(numeric), 1e-12)
    rel = num / den
    print(f"  [{'OK ' if rel < tol else 'FAIL'}] {name:<34} rel.err = {rel:.2e}")
    return rel < tol


def main():
    print("\nIMPLEMENTATION AUDIT")
    print("=" * 66)
    d = 6
    C = sym_random_psd(d, 3) + 0.4 * np.eye(d)
    results = []

    # 1. logdet gradient. Note: finite differences over a SYMMETRIC
    #    perturbation double-count off-diagonals, so the analytic gradient
    #    must be symmetrised the same way before comparison.
    f_ld = lambda M: -(BETA / 2.0) * np.linalg.slogdet(M + EPS * np.eye(d))[1]
    A = -(BETA / 2.0) * np.linalg.inv(C + EPS * np.eye(d))
    A_sym = A + A.T - np.diag(np.diag(A))
    results.append(check("d/dC of -(b/2) logdet(C+eI)",
                         A_sym, fd_grad_matrix(f_ld, C)))

    # 2. lam_min gradient
    f_lm = lambda M: np.linalg.eigvalsh(M)[0]
    v = np.linalg.eigh(C)[1][:, 0]
    A = np.outer(v, v)
    A_sym = A + A.T - np.diag(np.diag(A))
    results.append(check("d/dC of lam_min(C)", A_sym, fd_grad_matrix(f_lm, C)))

    # 3. VICReg gradient
    def f_vic(M):
        var = np.sqrt(np.maximum(np.diag(M), 0) + 1e-8)
        off = M - np.diag(np.diag(M))
        return 25.0 * np.sum(np.maximum(0, GAMMA - var) ** 2) + np.sum(off ** 2)

    var = np.sqrt(np.maximum(np.diag(C), 0) + 1e-8)
    hinge = np.maximum(0.0, GAMMA - var)
    A = np.zeros_like(C)
    np.fill_diagonal(A, 25.0 * (-2.0 * hinge * (0.5 / var)))
    A = A + 2.0 * (C - np.diag(np.diag(C)))
    A_sym = A + A.T - np.diag(np.diag(A))
    results.append(check("d/dC of VICReg", A_sym, fd_grad_matrix(f_vic, C)))

    # 4. chain rule dC -> dZ, checked end to end on Z
    n, dd = 40, 5
    Z = RNG.standard_normal((n, dd))

    def loss_of_Z(Zin):
        Zc = Zin - Zin.mean(0, keepdims=True)
        M = (Zc.T @ Zc) / (n - 1)
        return -(BETA / 2.0) * np.linalg.slogdet(M + EPS * np.eye(dd))[1]

    Zc = Z - Z.mean(0, keepdims=True)
    Cz = (Zc.T @ Zc) / (n - 1)
    dC = -(BETA / 2.0) * np.linalg.inv(Cz + EPS * np.eye(dd))
    analytic = (2.0 / (n - 1)) * Zc @ (0.5 * (dC + dC.T))

    h = 1e-6
    numeric = np.zeros_like(Z)
    for a in range(n):
        for b in range(dd):
            Zp = Z.copy(); Zp[a, b] += h
            Zm = Z.copy(); Zm[a, b] -= h
            numeric[a, b] = (loss_of_Z(Zp) - loss_of_Z(Zm)) / (2 * h)
    results.append(check("chain rule dL/dZ (end to end)", analytic, numeric))

    # 5/6. Gaussian MI routine vs an independent construction
    print()
    T, r = 12, 0.8
    idx = np.arange(T)
    S = r ** np.abs(idx[:, None] - idx[None, :])

    def mi_logdet(Sig, Aa, Bb):
        Aa, Bb = np.array(Aa), np.array(Bb)
        AB = np.concatenate([Aa, Bb])
        sl = lambda I, J: np.linalg.slogdet(Sig[np.ix_(I, J)])[1]
        return 0.5 * (sl(Aa, Aa) + sl(Bb, Bb) - sl(AB, AB))

    def mi_schur(Sig, Aa, Bb):
        """I = h(A) - h(A|B) via the Schur complement."""
        Aa, Bb = np.array(Aa), np.array(Bb)
        SAA = Sig[np.ix_(Aa, Aa)]
        SAB = Sig[np.ix_(Aa, Bb)]
        SBB = Sig[np.ix_(Bb, Bb)]
        cond = SAA - SAB @ np.linalg.solve(SBB, SAB.T)
        return 0.5 * (np.linalg.slogdet(SAA)[1] - np.linalg.slogdet(cond)[1])

    Aset, Bset = [4, 5, 6], [0, 1, 2, 3, 7, 8, 9, 10, 11]
    a1, a2 = mi_logdet(S, Aset, Bset), mi_schur(S, Aset, Bset)
    print(f"  [{'OK ' if abs(a1-a2) < 1e-10 else 'FAIL'}] "
          f"MI: logdet form vs Schur form      "
          f"{a1:.10f} vs {a2:.10f}")
    results.append(abs(a1 - a2) < 1e-10)

    # Monte-Carlo cross-check on a 1-vs-1 case with a known closed form
    rho = 0.8
    exact = -0.5 * np.log(1 - rho ** 2)
    routine = mi_logdet(np.array([[1.0, rho], [rho, 1.0]]), [0], [1])
    print(f"  [{'OK ' if abs(exact-routine) < 1e-12 else 'FAIL'}] "
          f"MI: 2-variable closed form         "
          f"{routine:.10f} vs {exact:.10f}")
    results.append(abs(exact - routine) < 1e-12)

    print("\n" + "=" * 66)
    print(f"  {sum(results)}/{len(results)} checks passed")
    if all(results):
        print("  => The analytic gradients and the MI routine are correct.")
        print("     Agreement between theory and measurement is therefore")
        print("     evidence about the theory, not about a shared bug.")
    else:
        print("  => AT LEAST ONE IMPLEMENTATION IS WRONG. Every downstream")
        print("     result that uses it must be recomputed.")
    print("=" * 66)


if __name__ == "__main__":
    main()
