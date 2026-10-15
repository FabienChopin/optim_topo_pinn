# Pourquoi SIMP reste gris sur la bande 1D, et comment obtenir un design 0/1.
#
#     min  J = int_{0.2}^{0.3} u^2 dx
#     s.c. -(A u')' = 1 sur [0, 1], u(0) = u(1) = 0, int rho = V0 = 0.5, A in {A2, A1}
#
# Solution exacte pour tout A : A u' = c - x, donc u ne dépend que de 1/A. Un mélange fin des
# deux matériaux (couches en série) a pour conductivité effective la moyenne harmonique :
#     1/A_mix(rho) = rho/A1 + (1 - rho)/A2      (loi "mix", seule loi réalisable en 1D)
# SIMP P = 3 donne au gris une conductivité bien supérieure (A(0.5) = 0.134 contre 0.020) :
# le gris est un matériau fictif, meilleur que tout mélange réel, que l'optimiseur exploite.
#
# Quatre calculs, tous avec le même évaluateur exact (grille de N cellules, interfaces placées
# exactement par fraction de cellule, ce qui est exact en 1D) :
#   1. SIMP, continuation P = 1 -> 3 puis P -> 100 : J, gris, J du design seuillé
#   2. Loi "mix" sans régularisation : l'optimum relaxé est déjà binaire
#   3. Designs binaires à K interfaces, positions optimisées (multistart) : optimum binaire global
#   4. Champ de phase sur grille (Van der Waals, EPS -> 3h) avec SIMP P = 1, P = 3 et "mix"
#
#     python band_1d_binaire.py      (quelques minutes, numpy + scipy)
import time
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A1, A2, V0 = 1.0, 0.01, 0.5
J_A, J_B = 0.2, 0.3
N = 1000
H = 1.0 / N
XC = (np.arange(N) + 0.5) * H
WB = ((XC >= J_A) & (XC <= J_B)).astype(float)
LAMBDA = 1e-2
OUT = "band_1d_binaire.png"

# ----------------------------------------------------------------------------
# Évaluateur exact et gradient (adjoint explicite)
# ----------------------------------------------------------------------------
def J_of_b(b, grad=False):
    """b = 1/A par cellule. J = int_band u^2 avec u = int_0^x (c - s) b ds, u(1) = 0."""
    q = H * b
    c = (XC * q).sum() / q.sum()
    f = (c - XC) * q
    u = np.cumsum(f) - 0.5 * f
    J = H * (WB * u * u).sum()
    if not grad:
        return J
    g = 2.0 * H * WB * u
    v = np.cumsum(g[::-1])[::-1] - 0.5 * g
    return J, H * (v * (c - XC) + (v * q).sum() * (XC - c) / q.sum())

def inv_A(r, law, p=1.0):
    """1/A(rho) et sa dérivée."""
    if law == "mix":
        return r / A1 + (1.0 - r) / A2, np.full_like(r, 1.0 / A1 - 1.0 / A2)
    A = A2 + (A1 - A2) * r ** p
    return 1.0 / A, -(A1 - A2) * p * r ** (p - 1.0) / A ** 2

J_REF = J_of_b(np.full(N, 1.0 / (A2 + (A1 - A2) * V0)))     # homogène, SIMP P = 1

def J_rel(r, law="mix", p=1.0):
    """J/J_ref. Pour un design binaire, la loi n'a pas d'importance (exact)."""
    return J_of_b(inv_A(r, law, p)[0]) / J_REF

def grey(r):
    return (4.0 * r * (1.0 - r)).mean()

def interfaces(r):
    rb = (r > 0.5).astype(int)
    return np.round(XC[np.nonzero(np.diff(rb))[0]] + 0.5 * H, 3)

# ----------------------------------------------------------------------------
# Optimisation sur grille : L-BFGS-B (bornes), lagrangien augmenté sur le volume
# ----------------------------------------------------------------------------
def optimize_grid(r0, law, p=1.0, lam=0.0, eps=1.0, iters=3000):
    """min J/J_ref + lam * int(eps r'^2 + r^2 (1 - r)^2 / eps)  s.c. int r = V0, 0 <= r <= 1."""
    r, mult, pen = r0.copy(), 0.0, 1e3
    def f(r):
        b, db = inv_A(r, law, p)
        J, dJ = J_of_b(b, True)
        F, G = J / J_REF, dJ * db / J_REF
        if lam > 0:
            d = np.diff(r) / H
            F += lam * (eps * H * (d * d).sum() + H * (r * r * (1 - r) ** 2).sum() / eps)
            gd = np.zeros_like(r)
            gd[:-1] -= 2.0 * eps * d
            gd[1:] += 2.0 * eps * d
            G = G + lam * (gd + H * 2.0 * r * (1 - r) * (1 - 2 * r) / eps)
        cv = r.mean() - V0
        return F + mult * cv + 0.5 * pen * cv * cv, G + (mult + pen * cv) / N
    for _ in range(8):
        r = minimize(f, r, jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * N,
                     options={"maxiter": iters, "maxfun": 2 * iters, "ftol": 1e-15, "gtol": 1e-12}).x
        mult += pen * (r.mean() - V0)
    return r

# ----------------------------------------------------------------------------
# Designs binaires à K interfaces (fraction de phase par cellule : exact en 1D)
# ----------------------------------------------------------------------------
def rho_from_interfaces(pos, s0):
    edges = np.concatenate([[0.0], np.sort(np.clip(pos, 0.0, 1.0)), [1.0]])
    lo, r = np.arange(N) * H, np.zeros(N)
    for k in range(len(edges) - 1):
        if (s0 + k) % 2 == 1:
            r += np.clip(np.minimum(lo + H, edges[k + 1]) - np.maximum(lo, edges[k]), 0.0, None) / H
    return r

def best_binary(K, n_starts=40, seed=1):
    rng, best = np.random.default_rng(seed), (np.inf, None, None)
    for s0 in (0, 1):
        cons = {"type": "eq", "fun": lambda q: rho_from_interfaces(q, s0).mean() - V0}
        for _ in range(n_starts):
            res = minimize(lambda q: J_rel(rho_from_interfaces(q, s0)), np.sort(rng.random(K)),
                           method="SLSQP", constraints=[cons], bounds=[(0.0, 1.0)] * K,
                           options={"maxiter": 300, "ftol": 1e-12})
            if abs(rho_from_interfaces(res.x, s0).mean() - V0) < 1e-4 and res.fun < best[0]:
                best = (res.fun, np.sort(res.x), s0)
    return best

# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("Conductivité du gris rho = 0.5 :  SIMP P = 1 : %.3f   P = 3 : %.3f   mélange réel : %.3f"
          % (A2 + (A1 - A2) * 0.5, A2 + (A1 - A2) * 0.125, 1.0 / inv_A(np.array(0.5), "mix")[0]))
    print("SIMP ne pénalise le gris (A_SIMP <= A_mix) près de rho = 1 que si P >= A1/A2 = %g\n" % (A1 / A2))

    print("1. SIMP, continuation sur P")
    r, designs = np.full(N, V0), {}
    for p in [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 30.0, 100.0]:
        r = optimize_grid(r, "simp", p)
        rb = (r > 0.5).astype(float)
        print(f"   P = {p:5g}   J/J_ref = {J_rel(r, 'simp', p):.4f}   gris = {grey(r):.3f}   "
              f"seuillé : J/J_ref = {J_rel(rb):.4f} (volume {rb.mean():.3f})")
        if p in (3.0, 100.0):
            designs[f"SIMP P = {p:g}"] = r

    print("\n2. Loi de mélange 1D, sans régularisation")
    r = optimize_grid(np.full(N, V0), "mix")
    designs["mix, sans régularisation"] = r
    print(f"   J/J_ref = {J_rel(r):.4f}   gris = {grey(r):.4f}   interfaces {interfaces(r)}")

    print("\n3. Designs binaires à K interfaces, positions optimisées")
    for K in range(1, 5):
        Jk, pos, s0 = best_binary(K)
        print(f"   K = {K}   J/J_ref = {Jk:.4f}   interfaces {np.round(pos, 3)}   rho(0) = {s0}")
        if K == 2:
            designs["binaire optimal (K = 2)"] = rho_from_interfaces(pos, s0)

    print(f"\n4. Champ de phase sur grille, LAMBDA = {LAMBDA:g}, EPS 0.1 -> 3h")
    for law, p in [("simp", 1.0), ("simp", 3.0), ("mix", 1.0)]:
        r, eps = np.full(N, V0), 0.1
        while eps > 2.9 * H:
            r = optimize_grid(r, law, p, lam=LAMBDA, eps=eps, iters=2000)
            eps *= 10 ** -0.25
        rb = (r > 0.5).astype(float)
        name = f"champ de phase, {law}" + (f" P = {p:g}" if law == "simp" else "")
        designs[name] = r
        print(f"   {name:28s} EPS = {eps / 10 ** -0.25:.4f}   J/J_ref = {J_rel(r, law, p):.4f}   "
              f"gris = {grey(r):.3f}   seuillé : {J_rel(rb):.4f} (volume {rb.mean():.3f})   "
              f"interfaces {interfaces(r)}")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    rr = np.linspace(0, 1, 201)
    for p in (1, 3, 10, 100):
        ax[0].semilogy(rr, A2 + (A1 - A2) * rr ** p, label=f"SIMP P = {p}")
    ax[0].semilogy(rr, 1.0 / inv_A(rr, "mix")[0], "k", lw=2.5, label="mélange réel (série)")
    ax[0].set_xlabel("rho")
    ax[0].set_title("A(rho) : au-dessus de la courbe noire, le gris est un matériau fictif")
    ax[0].legend(fontsize=8)
    for name, r in designs.items():
        ax[1].plot(XC, r, lw=2.5 if "binaire" in name else 1.2,
                   ls="--" if "binaire" in name else "-", label=name)
    ax[1].axvspan(J_A, J_B, color="tab:red", alpha=0.1)
    ax[1].set_xlabel("x")
    ax[1].set_title("rho final")
    ax[1].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=100)
    print(f"\nTemps total : {time.time() - t0:.0f} s   Figure : {OUT}")

if __name__ == "__main__":
    main()
