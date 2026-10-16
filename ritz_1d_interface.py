# Deep Ritz en 1D sur une interface nette, design fixé (pas d'optimisation de rho).
#
#     -(A u')' = 1 sur [0, 1],  u(0) = u(1) = 0
#     A = A_LEFT sur [0, X_I[, A_RIGHT sur ]X_I, 1]   (échelon net, X_I = 0.5)
#
# Forme faible : trouver u in H^1_0 tel que int A u' v' = int v pour tout v in H^1_0.
# A symétrique définie positive : équivalent à minimiser l'énergie
#     E(u) = int (1/2 A u'^2 - u) dx
# Un seul réseau, u = x (1 - x) N(x) (Dirichlet exact). Dérivée première seulement : la
# continuité du flux A u' à l'interface est une condition naturelle, rien à imposer.
# La méthode ne connaît pas l'interface : quadrature uniforme (Gauss composite dont aucune
# borne d'élément ne tombe en X_I), fixe ou retirée à chaque itération.
#
# Piège : sur une quadrature fixe, l'énergie discrète sum w (1/2 A u'^2 - u) n'est pas bornée
# inférieurement pour un réseau : u peut monter par marches entre les points (u' petit aux
# points, raide entre eux). C'est le moins cher côté isolant (A petit). L-BFGS trouve ces
# marches ; Adam les trouve aussi si le réseau est assez riche par rapport à la quadrature.
# Retirer les points à chaque itération (Gauss décalé au hasard, Monte-Carlo) les supprime.
#
# Solution exacte : A u' = c - x (flux linéaire), u(x) = int_0^x (c - s) / A ds, c tel que u(1) = 0.
# Identité de Ritz : E(u) - E(u*) = 1/2 int A (u' - u*')^2 (erreur en norme d'énergie).
#
#     python ritz_1d_interface.py
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Paramètres
# ----------------------------------------------------------------------------
SEED = 3
A_LEFT, A_RIGHT = 1.0, 0.01   # conducteur à gauche, isolant à droite
X_I = 0.5                     # interface (inconnue de la méthode)
WIDTH, DEPTH = 32, 3
ADAM_LR = 1e-3            # Adam, taux décroissant en cosinus jusqu'à 0
LBFGS_ITERS = 3000
N_EVAL = 20001                # grille uniforme d'évaluation de l'erreur
# Cas : (nom, quadrature, éléments ou points, points de Gauss par élément, itérations Adam, L-BFGS ensuite)
#   "gauss"       : Gauss composite fixe
#   "gauss_shift" : Gauss composite décalé d'une fraction aléatoire d'élément à chaque itération
#   "mc"          : points uniformes retirés à chaque itération (poids 1/N)
# L-BFGS seulement sur quadrature fixe (objectif déterministe).
CASES = [
    ("Gauss 49x6 fixe, Adam 3000 + L-BFGS", "gauss", 49, 6, 3000, True),
    ("Gauss 49x6 fixe, Adam 10000", "gauss", 49, 6, 10000, False),
    ("Gauss 199x6 fixe, Adam 10000", "gauss", 199, 6, 10000, False),
    ("Gauss 199x6 fixe, Adam 3000 + L-BFGS", "gauss", 199, 6, 3000, True),
    ("Gauss 49x6 décalé, Adam 10000", "gauss_shift", 49, 6, 10000, False),
    ("Gauss 199x6 décalé, Adam 10000", "gauss_shift", 199, 6, 10000, False),
    ("Monte-Carlo 300, Adam 10000", "mc", 300, None, 10000, False),
]
OUT = "ritz_1d_interface"

torch.set_default_dtype(torch.float64)

def conductivity(x):
    lib = torch if torch.is_tensor(x) else np
    return lib.where(x < X_I, A_LEFT, A_RIGHT) + 0.0 * x

# ----------------------------------------------------------------------------
# Solution exacte
# ----------------------------------------------------------------------------
C = (X_I ** 2 / 2 / A_LEFT + (1 - X_I ** 2) / 2 / A_RIGHT) / (X_I / A_LEFT + (1 - X_I) / A_RIGHT)

def exact_u(x):
    left = (C * x - x ** 2 / 2) / A_LEFT
    u_i = (C * X_I - X_I ** 2 / 2) / A_LEFT
    right = u_i + (C * (x - X_I) - (x ** 2 - X_I ** 2) / 2) / A_RIGHT
    return np.where(x < X_I, left, right)

def exact_du(x):
    return (C - x) / conductivity(x)

XE = np.linspace(0.0, 1.0, N_EVAL)
# grille d'évaluation : X_I est un noeud, intégrale par trapèzes exacte à O(h^2) de chaque côté
U_EX, DU_EX, A_EX = exact_u(XE), exact_du(XE), conductivity(XE)

def trapz(f):
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(XE)))

E_EX = trapz(0.5 * A_EX * DU_EX ** 2 - U_EX)
NORM_A_EX = np.sqrt(trapz(A_EX * DU_EX ** 2))

# ----------------------------------------------------------------------------
# Quadrature
# ----------------------------------------------------------------------------
def gauss_composite(n_el, n_g):
    t, wt = np.polynomial.legendre.leggauss(n_g)
    edges = np.linspace(0.0, 1.0, n_el + 1)
    x = (0.5 * (edges[:-1, None] + edges[1:, None]) + 0.5 * np.diff(edges)[:, None] * t).ravel()
    w = (0.5 * np.diff(edges)[:, None] * wt).ravel()
    return torch.tensor(x).reshape(-1, 1), torch.tensor(w).reshape(-1, 1)

# ----------------------------------------------------------------------------
# Réseau
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(1, WIDTH), nn.Tanh()]
        for _ in range(DEPTH - 1):
            layers += [nn.Linear(WIDTH, WIDTH), nn.Tanh()]
        layers.append(nn.Linear(WIDTH, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(2.0 * x - 1.0)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
    def forward(self, x):
        return x * (1.0 - x) * self.mlp(x)

def d_dx(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

def energy(u_model, X, W):
    x = X.clone().requires_grad_(True)
    u = u_model(x)
    du = d_dx(u, x)
    return (W * (0.5 * conductivity(x) * du ** 2 - u)).sum()

# ----------------------------------------------------------------------------
# Erreurs sur la grille fine (indépendantes de la quadrature d'entraînement)
# ----------------------------------------------------------------------------
def errors(u_model):
    x = torch.tensor(XE).reshape(-1, 1).requires_grad_(True)
    u = u_model(x)
    du = d_dx(u, x).detach().numpy().ravel()
    u = u.detach().numpy().ravel()
    flux = A_EX * du
    near = np.abs(XE - X_I) < 0.05
    e_density = A_EX * (du - DU_EX) ** 2
    return {
        "u": u, "du": du, "flux": flux,
        "L2": np.sqrt(trapz((u - U_EX) ** 2) / trapz(U_EX ** 2)),
        "energie": np.sqrt(trapz(A_EX * (du - DU_EX) ** 2)) / NORM_A_EX,
        "E-E*": trapz(0.5 * A_EX * du ** 2 - u) - E_EX,
        "flux_max": np.abs(flux - (C - XE)).max(),
        "flux_loin": np.abs(flux - (C - XE))[~near].max(),
        "part_pres": trapz(np.where(near, e_density, 0.0)) / trapz(e_density),
    }

# ----------------------------------------------------------------------------
# Entraînement
# ----------------------------------------------------------------------------
def train(kind, n, n_g, adam_iters, use_lbfgs):
    torch.manual_seed(SEED)
    u_model = UNet()
    if kind == "mc":
        sample = lambda: (torch.rand(n, 1), torch.full((n, 1), 1.0 / n))
    else:
        X, W = gauss_composite(n, n_g)
        if kind == "gauss":
            sample = lambda: (X, W)
        else:   # le décalage modulo 1 garde une quadrature exacte pour les fonctions 1-périodiques
            sample = lambda: ((X + torch.rand(1) / n) % 1.0, W)
    opt = torch.optim.Adam(u_model.parameters(), lr=ADAM_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, adam_iters)
    for _ in range(adam_iters):
        opt.zero_grad()
        loss = energy(u_model, *sample())
        loss.backward()
        opt.step()
        sched.step()
    if use_lbfgs:
        lbfgs = torch.optim.LBFGS(u_model.parameters(), lr=1.0, max_iter=LBFGS_ITERS,
                                  max_eval=int(1.25 * LBFGS_ITERS), history_size=50,
                                  tolerance_grad=1e-12, tolerance_change=1e-15,
                                  line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            loss = energy(u_model, X, W)
            loss.backward()
            return loss
        lbfgs.step(closure)
    return u_model

def main():
    print(f"A = {A_LEFT:g} sur [0, {X_I}[, {A_RIGHT:g} sur ]{X_I}, 1]   c = {C:.6f}   "
          f"u*(X_I) = {exact_u(np.array(X_I)):.4f}   max u* = {U_EX.max():.4f}   E* = {E_EX:.6e}")
    print(f"MLP {WIDTH} x {DEPTH}, Adam lr {ADAM_LR:g} (cosinus), L-BFGS {LBFGS_ITERS}\n")
    print("Erreur flux = |A u' - (c - x)| ; 'à ±0.05' = part de l'erreur d'énergie à moins de 0.05 de X_I\n")
    print(f"{'cas':40s} {'L2 rel u':>9s} {'énergie rel':>11s} {'E - E*':>9s} "
          f"{'flux max':>9s} {'flux loin':>9s} {'à ±0.05':>8s} {'temps':>6s}")
    results = []
    for name, kind, n, n_g, adam_iters, use_lbfgs in CASES:
        t0 = time.time()
        u_model = train(kind, n, n_g, adam_iters, use_lbfgs)
        ev = errors(u_model)
        results.append((name, ev))
        print(f"{name:40s} {ev['L2']:9.2e} {ev['energie']:11.2e} {ev['E-E*']:9.2e} "
              f"{ev['flux_max']:9.2e} {ev['flux_loin']:9.2e} {ev['part_pres']:8.2f} {time.time() - t0:5.1f}s",
              flush=True)
    draw(results)
    print(f"\nFigure : {OUT}.png")

def draw(results):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    ax[0].plot(XE, U_EX, "k", lw=2.5, label="u exact")
    ax[1].plot(XE, C - XE, "k", lw=2.5, label="flux exact c - x")
    results = [(name, ev) for name, ev in results if ev["L2"] < 1.0]   # cas divergents : tableau seulement
    for name, ev in results:
        ax[0].plot(XE, ev["u"], ls="--", label=name)
        ax[1].plot(XE, ev["flux"], ls="--", label=name)
        ax[2].semilogy(XE, np.abs(ev["u"] - U_EX) + 1e-16, lw=1, label=name)
    ax[0].set_title("u")
    ax[1].set_title("flux A u'")
    ax[2].set_title("|u - u*|")
    for a in ax:
        a.axvline(X_I, color="tab:red", alpha=0.4)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle(f"Deep Ritz, interface nette en x = {X_I}   A = {A_LEFT:g} | {A_RIGHT:g}   MLP {WIDTH} x {DEPTH}")
    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=100)
    plt.close(fig)

if __name__ == "__main__":
    main()
