# Deep Ritz en 1D sur une interface fixée, nette ou lissée (pas d'optimisation de rho).
#
#     -(A u')' = 1 sur [0, 1],  u(0) = u(1) = 0
#     A = A_LEFT sur [0, X_I[, A_RIGHT sur ]X_I, 1]   (marche nette, X_I = 0.5, INTERFACE_W = 0)
#     A = A_RIGHT + (A_LEFT - A_RIGHT) rho,  rho = sigmoid(-(x - X_I) / delta)   (lissée)
#     rho passe de 99 % à 1 % sur l'épaisseur INTERFACE_W = 2 ln(99) delta
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
# Intégrée sur une grille fine resserrée autour de X_I (résout l'épaisseur de l'interface).
# Identité de Ritz : E(u) - E(u*) = 1/2 int A (u' - u*')^2 (erreur en norme d'énergie).
#
# Interface lissée -> marche nette quand INTERFACE_W -> 0 : 1/A_w -> 1/A sauf sur une largeur
# ~ INTERFACE_W, où l'écart reste borné par 1/A_RIGHT. Donc u_w -> u en O(w), norme d'énergie
# en O(sqrt(w)). Le mode "epaisseur" le vérifie et relance Deep Ritz pour plusieurs épaisseurs.
#
#     python ritz_1d_interface.py             cas de CASES à l'épaisseur INTERFACE_W
#     python ritz_1d_interface.py epaisseur   balayage de l'épaisseur (WIDTHS_SCAN, CASES_SCAN)
import sys
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
INTERFACE_W = 0.0             # épaisseur 99 % -> 1 % de rho ; 0 : marche nette
WIDTH, DEPTH = 32, 3
ADAM_LR = 1e-3            # Adam, taux décroissant en cosinus jusqu'à 0
LBFGS_ITERS = 3000
N_EVAL = 20001                # grille uniforme d'évaluation, + N_CLUSTER noeuds de chaque côté de X_I
N_CLUSTER = 4000
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
WIDTHS_SCAN = [0.0, 1e-1, 1e-2, 1e-3, 1e-5]
CASES_SCAN = [CASES[0], CASES[5]]   # quadrature fixe + L-BFGS (marches), Gauss décalé (stable)
OUT = "ritz_1d_interface"

torch.set_default_dtype(torch.float64)

# ----------------------------------------------------------------------------
# Problème : conductivité et solution exacte
# ----------------------------------------------------------------------------
class Probleme:
    """Conductivité, solution exacte et grille d'évaluation pour une épaisseur d'interface w."""
    def __init__(self, w):
        self.w = w
        self.delta = w / (2.0 * np.log(99.0))
        # grille uniforme + noeuds géométriques de part et d'autre de X_I (X_I est un noeud)
        offsets = np.geomspace(1e-3 * self.delta if w > 0 else 1e-12, 0.05, N_CLUSTER)
        self.x = np.unique(np.concatenate([np.linspace(0.0, 1.0, N_EVAL), [X_I],
                                           X_I - offsets, X_I + offsets]))
        self.a = self.conductivity(self.x)
        inv_a = 1.0 / self.a
        self.c = self.trapz(self.x * inv_a) / self.trapz(inv_a)
        self.du = (self.c - self.x) / self.a
        self.u = self.cumtrapz(self.du)
        self.energy = self.trapz(0.5 * self.a * self.du ** 2 - self.u)
        self.norm_a = np.sqrt(self.trapz(self.a * self.du ** 2))

    def conductivity(self, x):
        lib = torch if torch.is_tensor(x) else np
        if self.w == 0.0:
            return lib.where(x < X_I, A_LEFT, A_RIGHT) + 0.0 * x
        rho = 0.5 * (1.0 - lib.tanh((x - X_I) / (2.0 * self.delta)))   # sigmoid(-(x - X_I) / delta), sans overflow
        return A_RIGHT + (A_LEFT - A_RIGHT) * rho

    def trapz(self, f):
        return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(self.x)))

    def cumtrapz(self, f):
        return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(self.x))])

def describe(pb):
    nature = "marche nette" if pb.w == 0.0 else f"sigmoïde, épaisseur 99 % -> 1 % = {pb.w:g}"
    return f"A = {A_LEFT:g} | {A_RIGHT:g} en x = {X_I} ({nature})"

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

def energy(u_model, X, W, pb):
    x = X.clone().requires_grad_(True)
    u = u_model(x)
    du = d_dx(u, x)
    return (W * (0.5 * pb.conductivity(x) * du ** 2 - u)).sum()

# ----------------------------------------------------------------------------
# Erreurs sur la grille fine (indépendantes de la quadrature d'entraînement)
# ----------------------------------------------------------------------------
def l2_rel(u, u_ref, pb):
    return np.sqrt(pb.trapz((u - u_ref) ** 2) / pb.trapz(u_ref ** 2))

def errors(u_model, pb):
    x = torch.tensor(pb.x).reshape(-1, 1).requires_grad_(True)
    u = u_model(x)
    du = d_dx(u, x).detach().numpy().ravel()
    u = u.detach().numpy().ravel()
    flux = pb.a * du
    near = np.abs(pb.x - X_I) < 0.05
    e_density = pb.a * (du - pb.du) ** 2
    return {
        "u": u, "du": du, "flux": flux,
        "L2": l2_rel(u, pb.u, pb),
        "energie": np.sqrt(pb.trapz(e_density)) / pb.norm_a,
        "E-E*": pb.trapz(0.5 * pb.a * du ** 2 - u) - pb.energy,
        "flux_max": np.abs(flux - (pb.c - pb.x)).max(),
        "flux_loin": np.abs(flux - (pb.c - pb.x))[~near].max(),
        "part_pres": pb.trapz(np.where(near, e_density, 0.0)) / pb.trapz(e_density),
    }

# ----------------------------------------------------------------------------
# Entraînement
# ----------------------------------------------------------------------------
def train(pb, kind, n, n_g, adam_iters, use_lbfgs):
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
        loss = energy(u_model, *sample(), pb)
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
            loss = energy(u_model, X, W, pb)
            loss.backward()
            return loss
        lbfgs.step(closure)
    return u_model

def main():
    pb = Probleme(INTERFACE_W)
    print(f"{describe(pb)}   c = {pb.c:.6f}   max u* = {pb.u.max():.4f}   E* = {pb.energy:.6e}")
    print(f"MLP {WIDTH} x {DEPTH}, Adam lr {ADAM_LR:g} (cosinus), L-BFGS {LBFGS_ITERS}\n")
    print("Erreur flux = |A u' - (c - x)| ; 'à ±0.05' = part de l'erreur d'énergie à moins de 0.05 de X_I\n")
    print(f"{'cas':40s} {'L2 rel u':>9s} {'énergie rel':>11s} {'E - E*':>9s} "
          f"{'flux max':>9s} {'flux loin':>9s} {'à ±0.05':>8s} {'temps':>6s}")
    results = []
    for name, kind, n, n_g, adam_iters, use_lbfgs in CASES:
        t0 = time.time()
        u_model = train(pb, kind, n, n_g, adam_iters, use_lbfgs)
        ev = errors(u_model, pb)
        results.append((name, ev))
        print(f"{name:40s} {ev['L2']:9.2e} {ev['energie']:11.2e} {ev['E-E*']:9.2e} "
              f"{ev['flux_max']:9.2e} {ev['flux_loin']:9.2e} {ev['part_pres']:8.2f} {time.time() - t0:5.1f}s",
              flush=True)
    draw(pb, results)
    print(f"\nFigure : {OUT}.png")

def scan_width():
    """Solution exacte lissée -> marche nette, puis Deep Ritz pour chaque épaisseur."""
    step = Probleme(0.0)
    print("1. Solution exacte lissée u_w comparée à la marche nette u_0 (sur la grille de u_w)\n")
    print(f"{'épaisseur':>10s} {'L2 rel':>9s} {'max |u_w - u_0|':>16s} {'énergie rel':>12s} {'E_w - E_0':>10s}")
    for w in WIDTHS_SCAN[1:]:
        pb = Probleme(w)
        u0 = np.interp(pb.x, step.x, step.u)
        a0 = step.conductivity(pb.x)
        du0 = (step.c - pb.x) / a0
        en = np.sqrt(pb.trapz(a0 * (pb.du - du0) ** 2)) / step.norm_a
        print(f"{w:10.0e} {l2_rel(pb.u, u0, pb):9.2e} {np.abs(pb.u - u0).max():16.2e} {en:12.2e} "
              f"{pb.energy - step.energy:10.2e}", flush=True)
    print("\n2. Deep Ritz : erreur par rapport à u_w (même épaisseur) et à u_0 (marche nette)\n")
    print(f"{'épaisseur':>10s} {'cas':40s} {'L2 / u_w':>9s} {'énergie / u_w':>13s} {'L2 / u_0':>9s} {'temps':>6s}")
    for w in WIDTHS_SCAN:
        pb = Probleme(w)
        for name, kind, n, n_g, adam_iters, use_lbfgs in CASES_SCAN:
            t0 = time.time()
            u_model = train(pb, kind, n, n_g, adam_iters, use_lbfgs)
            ev = errors(u_model, pb)
            print(f"{w:10.0e} {name:40s} {ev['L2']:9.2e} {ev['energie']:13.2e} "
                  f"{l2_rel(ev['u'], np.interp(pb.x, step.x, step.u), pb):9.2e} {time.time() - t0:5.1f}s",
                  flush=True)

def draw(pb, results):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    ax[0].plot(pb.x, pb.u, "k", lw=2.5, label="u exact")
    ax[1].plot(pb.x, pb.c - pb.x, "k", lw=2.5, label="flux exact c - x")
    results = [(name, ev) for name, ev in results if ev["L2"] < 1.0]   # cas divergents : tableau seulement
    for name, ev in results:
        ax[0].plot(pb.x, ev["u"], ls="--", label=name)
        ax[1].plot(pb.x, ev["flux"], ls="--", label=name)
        ax[2].semilogy(pb.x, np.abs(ev["u"] - pb.u) + 1e-16, lw=1, label=name)
    ax[0].set_title("u")
    ax[1].set_title("flux A u'")
    ax[2].set_title("|u - u*|")
    for a in ax:
        a.axvline(X_I, color="tab:red", alpha=0.4)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle(f"Deep Ritz, {describe(pb)}   MLP {WIDTH} x {DEPTH}")
    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=100)
    plt.close(fig)

if __name__ == "__main__":
    scan_width() if sys.argv[1:] == ["epaisseur"] else main()
