# Version 1D de l'optimisation par champ de phase et PINN (même méthode que le 2D),
# pour observer la forme de rho et l'effet de chaque paramètre en quelques secondes.
#
#     min  J = int_0^1 w(x) (u - u_target)^2 dx
#     s.c. -(A(rho) u')' = 1 sur [0, 1], u(0) = u(1) = 0, int rho = V0
#     A = A2 + (A1 - A2) rho^P,  rho = sigmoid(N / eps)
#
# Solution exacte pour tout rho : A u' = c - x, donc u(x) = int_0^x (c - s) / A ds avec c tel
# que u(1) = 0. Sert d'évaluateur exact (pas de solveur) et de référence SIMP.
#
# Problèmes (PROBLEM) :
#   "band"      : w = 1 sur [0.2, 0.3], u_target = 0 (le cas 2D réduit à 1D)
#   "two_bands" : w = 1 sur [0.2, 0.3] et [0.65, 0.75], u_target = 0
#   "tracking"  : w = 1 partout, u_target produit par un design binaire : fond conducteur et
#                 couches isolantes (TRACKING_LAYERS) d'épaisseurs variées, dont certaines plus fines
#                 que l'interface à grand EPS. Solution unique et connue (A u' = c - x fixe A là où
#                 u' != 0) : on mesure directement l'erreur sur rho.
#
#     python pinn_1d.py
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
PROBLEM = "tracking"
TRACKING_LAYERS = [(0.05, 0.15), (0.30, 0.34), (0.50, 0.70), (0.82, 0.85)]   # couches isolantes rho = 0
SEED = 3
A1, A2 = 1.0, 0.01
P_MAX, P_STEP = 3.0, 0.5
P_AFTER_EPS = False       # True : EPS descend d'abord à P = 1, puis P monte à EPS final
EPS0 = 0.1
EPS_FACTOR = 10 ** (-0.1)
N_EPS = 11                # paliers en EPS (0.1 -> 0.01)
RHO_SCALE_BY_EPS = True   # rho = sigmoid(N / eps) si True, sigmoid(N) si False
LAMBDA = 1e-2             # Van der Waals
MU = 100.0                # volume
BETA = 1e4                # résidu fort
N_ELEMENTS, N_GAUSS_PER_ELEMENT = 50, 6    # Gauss composite (300 points), bornes des bandes alignées
WIDTH, DEPTH = 32, 3
ADAM_ITERS, ADAM_LR = 1000, 1e-3
LBFGS_CHUNK = 300
MAX_CHUNKS, MAX_CHUNKS_RAMP = 4, 2
TOL_RHO, TOL_RES, TOL_STAGNATION = 1e-3, 1e-2, 1e-6
N_EXACT = 4000            # cellules de l'évaluateur exact
OUT = f"pinn_1d_{PROBLEM}_s{SEED}"

torch.set_default_dtype(torch.float64)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------------
# Problème : poids w, cible u_target, volume
# ----------------------------------------------------------------------------
def indicator(x, intervals):
    out = np.zeros_like(x, dtype=float)
    for a, b in intervals:
        out[(x >= a) & (x <= b)] = 1.0
    return out

def exact_u(rho, p, xc):
    """u aux centres xc (cellules de largeur h) pour la densité rho (numpy ou torch)."""
    lib = torch if torch.is_tensor(rho) else np
    h = 1.0 / len(xc)
    w = h / (A2 + (A1 - A2) * rho ** p)
    c = (xc * w).sum() / w.sum()
    return lib.cumsum((c - xc) * w, 0) - 0.5 * (c - xc) * w

XC = (np.arange(N_EXACT) + 0.5) / N_EXACT
if PROBLEM == "band":
    BANDS, V0, RHO_TRUE = [(0.2, 0.3)], 0.5, None
elif PROBLEM == "two_bands":
    BANDS, V0, RHO_TRUE = [(0.2, 0.3), (0.65, 0.75)], 0.5, None
elif PROBLEM == "tracking":
    BANDS = [(0.0, 1.0)]
    RHO_TRUE = 1.0 - indicator(XC, TRACKING_LAYERS)
    V0 = float(RHO_TRUE.mean())
else:
    raise ValueError(PROBLEM)
W_EXACT = indicator(XC, BANDS)
U_TARGET_EXACT = exact_u(RHO_TRUE, 1.0, XC) if RHO_TRUE is not None else np.zeros(N_EXACT)

def exact_J(rho, p):
    u = exact_u(rho, p, XC if not torch.is_tensor(rho) else torch.tensor(XC))
    ut = U_TARGET_EXACT if not torch.is_tensor(rho) else torch.tensor(U_TARGET_EXACT)
    w = W_EXACT if not torch.is_tensor(rho) else torch.tensor(W_EXACT)
    return (w * (u - ut) ** 2).sum() / N_EXACT

J_REF = float(exact_J(np.full(N_EXACT, V0), 1.0))     # mélange homogène rho = V0, P = 1

# ----------------------------------------------------------------------------
# Référence SIMP 1D (gradient exact par autograd sur la solution exacte, volume exact)
# ----------------------------------------------------------------------------
def simp_reference(n=1000, iters_per_p=300):
    xc = torch.tensor((np.arange(n) + 0.5) / n)
    w = torch.tensor(np.interp((np.arange(n) + 0.5) / n, XC, W_EXACT))
    ut = torch.tensor(np.interp((np.arange(n) + 0.5) / n, XC, U_TARGET_EXACT))
    rho = torch.full((n,), V0)
    p_values = list(np.arange(1.0, P_MAX + 1e-9, P_STEP))
    for p in p_values:
        for _ in range(iters_per_p):
            r = rho.clone().requires_grad_(True)
            J = (w * (exact_u(r, p, xc) - ut) ** 2).sum() / n
            J.backward()
            g = r.grad
            target = rho - 0.05 * g / g.abs().max()
            lo, hi = -1.0, 1.0
            for _ in range(60):
                eta = 0.5 * (lo + hi)
                new = (target + eta).clamp(0.0, 1.0)
                lo, hi = (lo, eta) if new.mean() > V0 else (eta, hi)
            rho = new
    rho = np.interp(XC, xc.numpy(), rho.numpy())
    return rho, float(exact_J(rho, P_MAX)) / J_REF

# ----------------------------------------------------------------------------
# Quadrature de Gauss composite
# ----------------------------------------------------------------------------
t, wt = np.polynomial.legendre.leggauss(N_GAUSS_PER_ELEMENT)
edges = np.linspace(0.0, 1.0, N_ELEMENTS + 1)
x_np = (0.5 * (edges[:-1, None] + edges[1:, None]) + 0.5 * np.diff(edges)[:, None] * t).ravel()
w_np = (0.5 * np.diff(edges)[:, None] * wt).ravel()
X = torch.tensor(x_np).reshape(-1, 1)
W = torch.tensor(w_np).reshape(-1, 1)
WJ = torch.tensor(indicator(x_np, BANDS)).reshape(-1, 1)
UT = torch.tensor(np.interp(x_np, XC, U_TARGET_EXACT)).reshape(-1, 1)

# ----------------------------------------------------------------------------
# Réseaux, opérateurs, losses
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

class RhoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.eps = EPS0
        scale = EPS0 if RHO_SCALE_BY_EPS else 1.0
        last = self.mlp.net[-1]
        with torch.no_grad():
            last.weight.mul_(0.1 * scale)
            last.bias.fill_(float(scale * np.log(V0 / (1.0 - V0))))
    def forward(self, x):
        n = self.mlp(x)
        return torch.sigmoid(n / self.eps if RHO_SCALE_BY_EPS else n)

def d_dx(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

def conductivity(rho, p):
    return A2 + (A1 - A2) * rho ** p

def compute_losses(u_model, rho_model, eps, p):
    x = X.clone().requires_grad_(True)
    u = u_model(x)
    rho = rho_model(x)
    r = -d_dx(conductivity(rho, p) * d_dx(u, x), x) - 1.0
    L_res = (W * r ** 2).sum()
    drho = d_dx(rho, x)
    L_vdw = (W * (eps * drho ** 2 + rho ** 2 * (1.0 - rho) ** 2 / eps)).sum()
    vol = (W * rho).sum()
    J = (W * WJ * (u - UT) ** 2).sum()
    total = J / J_REF + BETA * L_res + LAMBDA * L_vdw + MU * (vol - V0) ** 2
    parts = {"total": total.item(), "J/J_ref": (J / J_REF).item(), "residual": L_res.item(),
             "vdw": L_vdw.item(), "volume": vol.item(),
             "grey": (W * 4.0 * rho * (1.0 - rho)).sum().item()}
    return total, parts

# ----------------------------------------------------------------------------
# Évaluation exacte du design du PINN
# ----------------------------------------------------------------------------
def evaluate(rho_model, p):
    with torch.no_grad():
        rho = rho_model(torch.tensor(XC).reshape(-1, 1)).numpy().ravel()
    binary = (rho > 0.5).astype(float)
    out = {"rho": rho, "J_exact": float(exact_J(rho, p)) / J_REF,
           "J_binary": float(exact_J(binary, p)) / J_REF, "vol_binary": float(binary.mean())}
    if RHO_TRUE is not None:
        out["rho_error"] = float(np.abs(rho - RHO_TRUE).mean())   # aire mal placée
    return out

# ----------------------------------------------------------------------------
# Entraînement : continuation sur EPS et P, paliers arrêtés sur stabilité
# ----------------------------------------------------------------------------
def schedule(i):
    if P_AFTER_EPS:
        eps = EPS0 * EPS_FACTOR ** min(i, N_EPS - 1)
        p = 1.0 if i < N_EPS else min(P_MAX, 1.0 + P_STEP * (i - N_EPS + 1))
    else:
        eps = EPS0 * EPS_FACTOR ** i
        p = min(P_MAX, 1.0 + P_STEP * i)
    return eps, p

n_ramp = round((P_MAX - 1.0) / P_STEP)
N_CYCLES = N_EPS + n_ramp if P_AFTER_EPS else N_EPS

def main():
    t_start = time.time()
    print(f"PROBLEM = {PROBLEM}   V0 = {V0:.3f}   J_ref = {J_REF:.4e}   SEED = {SEED}   "
          f"P_AFTER_EPS = {P_AFTER_EPS}   RHO_SCALE_BY_EPS = {RHO_SCALE_BY_EPS}   "
          f"LAMBDA = {LAMBDA:g}   BETA = {BETA:g}")
    rho_simp, J_simp = simp_reference()
    print(f"Référence SIMP 1D : J/J_ref = {J_simp:.4f}")
    u_model, rho_model = UNet(), RhoNet()
    params = list(u_model.parameters()) + list(rho_model.parameters())
    snapshots = []
    for i in range(N_CYCLES):
        eps, p = schedule(i)
        rho_model.eps = eps
        t0 = time.time()
        if i == 0:
            opt = torch.optim.Adam(params, lr=ADAM_LR)
            for _ in range(ADAM_ITERS):
                opt.zero_grad()
                total, _ = compute_losses(u_model, rho_model, eps, p)
                total.backward()
                opt.step()
        lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=LBFGS_CHUNK, max_eval=int(1.25 * LBFGS_CHUNK),
                                  history_size=50, tolerance_grad=1e-11, tolerance_change=1e-14,
                                  line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            total, _ = compute_losses(u_model, rho_model, eps, p)
            total.backward()
            return total
        with torch.no_grad():
            rho_old = rho_model(X)
        loss_old = compute_losses(u_model, rho_model, eps, p)[1]["total"]
        max_chunks = MAX_CHUNKS if p >= P_MAX else MAX_CHUNKS_RAMP
        status = "plafond"
        for k in range(1, max_chunks + 1):
            lbfgs.step(closure)
            with torch.no_grad():
                rho_new = rho_model(X)
            d_rho = (W * (rho_new - rho_old).abs()).sum().item()
            rho_old = rho_new
            _, cur = compute_losses(u_model, rho_model, eps, p)
            d_loss = abs(cur["total"] - loss_old) / abs(loss_old)
            loss_old = cur["total"]
            if d_rho < TOL_RHO and np.sqrt(cur["residual"]) < TOL_RES:
                status = "stable"
                break
            if d_loss < TOL_STAGNATION:
                status = "stagnation"
                break
        ev = evaluate(rho_model, p)
        snapshots.append((eps, p, ev["rho"]))
        line = (f"Cycle {i:2d}  EPS = {eps:.3f}  P = {p:g}  [{status}, {k} tronçons, {time.time() - t0:4.1f} s]  "
                f"J_PINN/J_ref = {cur['J/J_ref']:.4f}  J_exact/J_ref = {ev['J_exact']:.4f}  "
                f"seuillé = {ev['J_binary']:.4f} (vol {ev['vol_binary']:.3f})  gris = {cur['grey']:.3f}  "
                f"vol = {cur['volume']:.3f}  ||r|| = {np.sqrt(cur['residual']):.1e}")
        if "rho_error" in ev:
            line += f"  int|rho - rho_vrai| = {ev['rho_error']:.3f}"
        print(line, flush=True)
    draw(snapshots, rho_simp, J_simp, u_model, rho_model, p)
    print(f"Temps total : {time.time() - t_start:.1f} s   Figure : {OUT}.png")

def draw(snapshots, rho_simp, J_simp, u_model, rho_model, p):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    cmap = plt.get_cmap("viridis")
    for k, (eps, p_k, rho) in enumerate(snapshots):
        ax[0].plot(XC, rho, color=cmap(k / max(1, len(snapshots) - 1)), lw=1.0,
                   label=f"ε = {eps:.3f}, P = {p_k:g}")
    ax[0].plot(XC, rho_simp, "k--", lw=1.2, label=f"SIMP (J/J_ref = {J_simp:.3f})")
    if RHO_TRUE is not None:
        ax[0].fill_between(XC, 0, RHO_TRUE, color="tab:orange", alpha=0.25, label="rho vrai")
    for a, b in BANDS if PROBLEM != "tracking" else []:
        ax[0].axvspan(a, b, color="tab:red", alpha=0.1)
    ax[0].set_title("rho à la fin de chaque palier")
    ax[0].set_xlabel("x")
    ax[0].legend(fontsize=7, loc="best")
    rho_final = snapshots[-1][2]
    xt = torch.tensor(XC).reshape(-1, 1)
    with torch.no_grad():
        u_pinn = u_model(xt).numpy().ravel()
    ax[1].plot(XC, exact_u(rho_final, p, XC), "k", lw=2, label="u exact (rho du PINN)")
    ax[1].plot(XC, u_pinn, "tab:red", ls="--", label="u PINN")
    if RHO_TRUE is not None:
        ax[1].plot(XC, U_TARGET_EXACT, "tab:orange", ls=":", lw=2, label="u cible")
    ax[1].legend(fontsize=8)
    ax[1].set_title("u (dernier palier)")
    ax[2].plot(XC, np.log10(conductivity(rho_final, p)), "tab:purple", label="PINN")
    ax[2].plot(XC, np.log10(conductivity(rho_simp, P_MAX)), "k--", label="SIMP")
    ax[2].set_ylim(np.log10(A2) - 0.1, 0.1)
    ax[2].set_title("log10 A (dernier palier)")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"{PROBLEM}   SEED = {SEED}   λ = {LAMBDA:g}   β = {BETA:g}   P après ε : {P_AFTER_EPS}   "
                 f"rho = sigmoid(N{'/ε' if RHO_SCALE_BY_EPS else ''})")
    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=100)
    plt.close(fig)

if __name__ == "__main__":
    main()
