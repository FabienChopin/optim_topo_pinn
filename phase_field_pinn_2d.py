# Optimisation topologique 2D par champ de phase, PINN en résidu fort,
# minimisation jointe (approche "all-at-once" par pénalité).
#
# Problème
# --------
#     min_{u, rho}   J(u) = int_{[0.2, 0.3] x [0, 1]} u^2 dx
#     s.c.           -div(A(rho) grad u) = 1   sur Omega = [0, 1]^2,  u = 0 sur le bord
#                    int_Omega rho dx = V0  (pénalisé)
#     A(rho) = A1*rho + A2*(1 - rho),  rho in [0, 1]
#
# Loss totale (minimisée conjointement sur les deux réseaux)
#     L = J / J_ref
#       + beta   * int (-div(A grad u) - 1)^2 dx                      (résidu fort, L2^2)
#       + lambda * int (eps |grad rho|^2 + rho^2 (1-rho)^2 / eps) dx  (Van der Waals)
#       + mu     * (int rho dx - V0)^2                                (volume)
#
# J_ref = J de la solution homogène rho = V0, calculée par série de Fourier
# (solution de -a Lap u = 1 sur le carré unité, u = 0 au bord).
#
# Discrétisation
#     - Gauss-Legendre tensoriel N_GAUSS x N_GAUSS sur [0, 1]^2 (résidu, VdW, volume)
#     - Gauss-Legendre tensoriel dédié sur la bande [J_A, J_B] x [0, 1] (objectif J)
#     - Strong BC : u(x, y) = x (1 - x) y (1 - y) N_u(x, y)
#     - rho(x, y) = sigmoid(N_rho(x, y))
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
# ----------------------------------------------------------------------------
# Paramètres
# ----------------------------------------------------------------------------
SEED = 3                  # graine : lancer plusieurs graines et garder le meilleur J
torch.set_default_dtype(torch.float64)
_ = torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A1, A2 = 1.0, 0.01        # conductivités des deux phases
P = 1.0                   # A(rho) = A2 + (A1 - A2) * rho**P   (P = 1 : interpolation linéaire)
V0 = 0.5                  # fraction volumique cible
EPS = 0.1                 # épaisseur d'interface initiale
LAMBDA = 1e-3             # poids Van der Waals
MU = 100.0                # pénalisation du volume
BETA = 1e4                # poids du résidu fort (penalty all-at-once)
N_GAUSS = 300             # points de Gauss par direction sur [0, 1]^2 (-> 90 000 points)
N_GAUSS_JX = 40           # points de Gauss en x sur [J_A, J_B]
N_GAUSS_JY = 200          # points de Gauss en y sur [0, 1]
J_A, J_B = 0.2, 0.3       # bande de l'objectif : [J_A, J_B] x [0, 1]
WIDTH, DEPTH = 32, 3      # MLP : DEPTH couches cachées de WIDTH neurones
ADAM_ITERS = 2000         # Adam au premier palier seulement
ADAM_LR = 1e-3
LOG_EVERY = 500
N_CYCLES = 21             # paliers de continuation sur EPS
EPS_FACTOR = 10 ** (-0.1) # EPS <- EPS * EPS_FACTOR après chaque palier (0.1 -> 0.001)
# Arrêt de chaque palier sur stabilité : L-BFGS par tronçons, contrôle entre tronçons
LBFGS_CHUNK = 1500         # itérations L-BFGS par tronçon
MAX_CHUNKS = 20           # plafond par palier (-> 10 000 itérations au plus)
TOL_RHO, TOL_RES = 1e-3, 1e-2              # paliers intermédiaires : aire modifiée, ||r||_L2
TOL_RHO_FINAL, TOL_RES_FINAL = 2e-4, 2e-3  # dernier palier
TOL_STAGNATION = 1e-6     # variation relative de la loss sur un tronçon en dessous de laquelle on s'arrête
N_PLOT = 201              # grille de post-traitement N_PLOT x N_PLOT
OUT_FIG = f"resultats_phase_field_pinn_2d_s{SEED}.png"
OUT_MODELS = f"modeles_phase_field_pinn_2d_s{SEED}.pt"
N_FD = 499                # noeuds intérieurs par direction pour la vérification DF
OUT_FIG_FD = f"comparaison_u_DF_s{SEED}.png"
LOG_FILE = f"log_phase_field_pinn_2d_s{SEED}.txt"   # copie de toute la sortie console
# ----------------------------------------------------------------------------
# Journal : tout ce qui s'affiche (print, erreurs) est aussi écrit dans LOG_FILE
# ----------------------------------------------------------------------------
class Tee:
    """Duplique un flux de sortie dans un fichier, vidé à chaque écriture."""
    def __init__(self, stream, log_file):
        self.stream = stream
        self.tee_file = log_file
    def write(self, data):
        self.stream.write(data)
        self.tee_file.write(data)
        self.tee_file.flush()
    def flush(self):
        self.stream.flush()
        self.tee_file.flush()

while hasattr(sys.stdout, "tee_file"):   # script recollé : on revient aux flux d'origine
    sys.stdout = sys.stdout.stream

while hasattr(sys.stderr, "tee_file"):
    sys.stderr = sys.stderr.stream

_log_handle = open(LOG_FILE, "a", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log_handle)
sys.stderr = Tee(sys.stderr, _log_handle)
print(f"\n########## Lancement {time.strftime('%Y-%m-%d %H:%M:%S')}   SEED = {SEED} ##########")
print(f"P = {P}  V0 = {V0}  EPS0 = {EPS}  LAMBDA = {LAMBDA}  MU = {MU}  BETA = {BETA}  "
      f"N_GAUSS = {N_GAUSS}  WIDTH x DEPTH = {WIDTH} x {DEPTH}  N_CYCLES = {N_CYCLES}")
# ----------------------------------------------------------------------------
# Quadrature de Gauss-Legendre (tensorielle)
# ----------------------------------------------------------------------------
def gauss_legendre_1d(n, a, b):
    t, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * t + 0.5 * (b + a), 0.5 * (b - a) * w

def gauss_legendre_2d(nx, ny, xa, xb, ya, yb):
    """Points (N, 2) et poids (N, 1) de Gauss-Legendre tensoriel sur [xa, xb] x [ya, yb]."""
    x, wx = gauss_legendre_1d(nx, xa, xb)
    y, wy = gauss_legendre_1d(ny, ya, yb)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    w = np.outer(wx, wy).reshape(-1, 1)
    return pts, w

_X_np, _W_np = gauss_legendre_2d(N_GAUSS, N_GAUSS, 0.0, 1.0, 0.0, 1.0)
_XJ_np, _WJ_np = gauss_legendre_2d(N_GAUSS_JX, N_GAUSS_JY, J_A, J_B, 0.0, 1.0)
X, W = torch.tensor(_X_np, device=DEVICE), torch.tensor(_W_np, device=DEVICE)
XJ, WJ = torch.tensor(_XJ_np, device=DEVICE), torch.tensor(_WJ_np, device=DEVICE)
# ----------------------------------------------------------------------------
# Normalisation de J : solution homogène rho = V0 (série de Fourier)
# ----------------------------------------------------------------------------
def u_homogeneous(pts, a, n_terms=200):
    """-a Lap u = 1 sur [0,1]^2, u = 0 au bord :
    u = (1/a) sum_{m,n impairs} 16 / (pi^4 m n (m^2 + n^2)) sin(m pi x) sin(n pi y)."""
    k = np.arange(1, 2 * n_terms, 2)
    sx = np.sin(np.pi * np.outer(pts[:, 0], k))
    sy = np.sin(np.pi * np.outer(pts[:, 1], k))
    m, n = np.meshgrid(k, k, indexing="ij")
    c = 16.0 / (np.pi ** 4 * m * n * (m ** 2 + n ** 2))
    return ((sx @ c) * sy).sum(axis=1) / a

_a_hom = A2 + (A1 - A2) * V0 ** P
J_REF = float((_WJ_np.ravel() * u_homogeneous(_XJ_np, _a_hom) ** 2).sum())
# ----------------------------------------------------------------------------
# Réseaux
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=2, width=WIDTH, depth=DEPTH):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(2.0 * x - 1.0)  # entrées ramenées sur [-1, 1]^2

class UNet(nn.Module):
    """u = x (1 - x) y (1 - y) N(x, y) : Dirichlet homogène exact sur tout le bord."""
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
    def forward(self, x):
        bx, by = x[:, 0:1], x[:, 1:2]
        return bx * (1.0 - bx) * by * (1.0 - by) * self.mlp(x)

class RhoNet(nn.Module):
    """rho = sigmoid(N(x, y)) in (0, 1), initialisé proche de V0."""
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        last = self.mlp.net[-1]
        with torch.no_grad():
            last.weight.mul_(0.1)
            last.bias.fill_(float(np.log(V0 / (1.0 - V0))))  # logit(V0)
    def forward(self, x):
        return torch.sigmoid(self.mlp(x))

u_model = UNet().to(DEVICE)
rho_model = RhoNet().to(DEVICE)
params = list(u_model.parameters()) + list(rho_model.parameters())
# ----------------------------------------------------------------------------
# Opérateurs différentiels et losses
# ----------------------------------------------------------------------------
def grad(y, x):
    """Gradient de y (N, 1) par rapport à x (N, d) -> (N, d). Points indépendants."""
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

def divergence(q, x):
    """Divergence d'un champ q (N, d) -> (N, 1)."""
    return sum(grad(q[:, k:k + 1], x)[:, k:k + 1] for k in range(x.shape[1]))

def conductivity(rho):
    return A2 + (A1 - A2) * rho ** P

def compute_losses():
    x = X.clone().requires_grad_(True)
    u = u_model(x)
    gu = grad(u, x)                                   # (N, 2)
    rho = rho_model(x)
    grho = grad(rho, x)                               # (N, 2)
    # Résidu fort : autograd sur le flux q = A(rho) grad u
    q = conductivity(rho) * gu
    r = -divergence(q, x) - 1.0
    L_res = (W * r ** 2).sum()                        # ||r||_{L2}^2
    # Van der Waals
    L_vdw = (W * (EPS * (grho ** 2).sum(dim=1, keepdim=True)
                  + rho ** 2 * (1.0 - rho) ** 2 / EPS)).sum()
    # Volume
    vol = (W * rho).sum()
    L_vol = (vol - V0) ** 2
    # Objectif sur la bande [J_A, J_B] x [0, 1] (jeu de points dédié)
    J = (WJ * u_model(XJ) ** 2).sum()
    total = J / J_REF + BETA * L_res + LAMBDA * L_vdw + MU * L_vol
    parts = {
        "total": total.item(),
        "J/J_ref": (J / J_REF).item(),
        "residu": L_res.item(),
        "vdw": L_vdw.item(),
        "vol_pen": L_vol.item(),
        "volume": vol.item(),
    }
    return total, parts

history = {k: [] for k in ["total", "J/J_ref", "residu", "vdw", "vol_pen", "volume"]}
adam_ends = []   # indices de fin d'Adam (un par cycle), pour les figures
def record(parts):
    for k in history:
        history[k].append(parts[k])

def log(tag, it, parts):
    print(f"[{tag} {it:6d}] total={parts['total']:.4e}  J/Jref={parts['J/J_ref']:.4e}  "
          f"res={parts['residu']:.3e}  vdw={parts['vdw']:.3e}  "
          f"vol={parts['volume']:.4f}")

# ----------------------------------------------------------------------------
# Post-traitement
# ----------------------------------------------------------------------------
def draw_figure(i, rho_img, u_img, r_img):
    fig, axd = plt.subplot_mosaic([["rho", "u", "res"], ["loss", "loss", "vol"]], figsize=(18, 10))
    def show(a, img, title, extend="neither", **kw):
        im = a.imshow(img, origin="lower", extent=(0, 1, 0, 1), **kw)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, extend=extend)
        for xb in (J_A, J_B):
            a.axvline(xb, color="tab:orange", ls="--", lw=1.2)
        a.set_title(title)
        a.set_xlabel("x")
        a.set_ylabel("y")
    show(axd["rho"], rho_img, r"$\rho(x, y)$ (bande de J en pointillés)",
         cmap="gray_r", vmin=0.0, vmax=1.0)
    show(axd["u"], u_img, r"$u(x, y)$", cmap="viridis")
    # Résidu en log10|r|, échelle bornée aux percentiles 1-99 % (points aberrants saturés)
    log_r = np.log10(np.maximum(np.abs(r_img), 1e-16))
    r_lo, r_hi = np.percentile(log_r, [1, 99])
    show(axd["res"], log_r, r"$\log_{10}|-\nabla\cdot(A\nabla u) - 1|$ (percentiles 1-99 %)",
         extend="both", cmap="inferno", vmin=r_lo, vmax=r_hi)
    # Losses : loss totale et contributions pondérées, sur un seul axe
    it_axis = np.arange(len(history["total"]))
    ax_l = axd["loss"]
    total_hist = np.abs(np.asarray(history["total"]))
    ax_l.semilogy(it_axis, total_hist, color="k", lw=2.0, label="loss totale")
    sub = [("J/J_ref", 1.0, "J / J_ref", "tab:blue"),
           ("residu", BETA, f"β · résidu   (β = {BETA:g})", "tab:red"),
           ("vdw", LAMBDA, f"λ · Van der Waals   (λ = {LAMBDA:g})", "tab:green"),
           ("vol_pen", MU, f"μ · volume   (μ = {MU:g})", "tab:purple")]
    for key, weight, label, color in sub:
        y = weight * np.abs(np.asarray(history[key]))   # contribution pondérée à la loss totale
        ax_l.semilogy(it_axis, y, color=color, lw=1.0, alpha=0.8, label=label)
    # Bas de l'axe : PLOT_DYN_RANGE x (plus petite loss totale). En dessous, une
    # contribution est négligeable devant la totale : on la laisse sortir du cadre.
    PLOT_DYN_RANGE = 1e-6
    y_lo = PLOT_DYN_RANGE * total_hist[total_hist > 0].min()
    y_hi = 2.0 * max(total_hist.max(),
                     max(w * np.abs(np.asarray(history[k])).max() for k, w, _, _ in sub))
    ax_l.set_ylim(y_lo, y_hi)
    ax_l.set_ylabel("loss (contributions pondérées)")
    ax_l.set_xlabel("itérations (Adam | L-BFGS évaluations)")
    for n in adam_ends:
        ax_l.axvline(n, color="gray", ls="--", lw=1)
    ax_l.legend(fontsize=9, loc="best")
    ax_l.set_title("Losses")
    ax_l.grid(alpha=0.3)
    axv = axd["vol"]
    axv.plot(it_axis, history["volume"])
    axv.axhline(V0, color="k", ls="--", lw=1)
    for n in adam_ends:
        axv.axvline(n, color="gray", ls="--", lw=1)
    axv.set_title(r"$\int \rho\,dx$")
    axv.grid(alpha=0.3)
    fig.tight_layout()
    fname = f"{i}_{OUT_FIG}"
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    return fname

def plot_results(i):
    g = np.linspace(0.0, 1.0, N_PLOT)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    xp = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1),
                      device=DEVICE).requires_grad_(True)
    up = u_model(xp)
    rhop = rho_model(xp)
    rp = -divergence(conductivity(rhop) * grad(up, xp), xp) - 1.0
    def to_img(t):  # (N, 1) -> image (y, x) pour imshow(origin="lower")
        return t.detach().cpu().numpy().reshape(N_PLOT, N_PLOT).T
    return draw_figure(i, to_img(rhop), to_img(up), to_img(rp))

# ----------------------------------------------------------------------------
# Vérification : u par différences finies (rho du réseau figé) vs u du réseau
# ----------------------------------------------------------------------------
def solve_fd(rho_grid, g):
    """-div(A(rho) grad u) = 1 sur [0,1]^2, u = 0 au bord. Volumes finis 5 points,
    moyenne harmonique de A aux faces. rho_grid : (n+2, n+2) sur la grille g, bords compris."""
    n = len(g) - 2
    h = g[1] - g[0]
    A = conductivity(rho_grid)
    ax = 2.0 * A[:-1, :] * A[1:, :] / (A[:-1, :] + A[1:, :])   # faces entre i et i+1
    ay = 2.0 * A[:, :-1] * A[:, 1:] / (A[:, :-1] + A[:, 1:])   # faces entre j et j+1
    I, Jj = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
    k = (I - 1) * n + (Jj - 1)
    aE, aW, aN, aS = ax[I, Jj], ax[I - 1, Jj], ay[I, Jj], ay[I, Jj - 1]
    rows, cols, vals = [k.ravel()], [k.ravel()], [((aE + aW + aN + aS) / h ** 2).ravel()]
    for a, di, dj in ((aE, 1, 0), (aW, -1, 0), (aN, 0, 1), (aS, 0, -1)):
        I2, J2 = I + di, Jj + dj
        inside = (I2 >= 1) & (I2 <= n) & (J2 >= 1) & (J2 <= n)
        rows.append(k[inside])
        cols.append(((I2 - 1) * n + (J2 - 1))[inside])
        vals.append(-a[inside] / h ** 2)
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n * n, n * n))
    u = np.zeros((n + 2, n + 2))
    u[1:-1, 1:-1] = spla.spsolve(M, np.ones(n * n)).reshape(n, n)
    return u

def draw_fd_figure(i, g, u_nn, u_fd, J_nn, J_fd, rel_l2):
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.2))
    u_min, u_max = min(u_nn.min(), u_fd.min()), max(u_nn.max(), u_fd.max())
    def show(a, img, title, **kw):
        im = a.imshow(img.T, origin="lower", extent=(0, 1, 0, 1), **kw)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, extend=kw.pop("extend", "neither"))
        for xb in (J_A, J_B):
            a.axvline(xb, color="tab:orange", ls="--", lw=1.2)
        a.set_title(title)
        a.set_xlabel("x")
        a.set_ylabel("y")
    show(ax[0], u_nn, r"$u$ PINN", cmap="viridis", vmin=u_min, vmax=u_max)
    show(ax[1], u_fd, r"$u$ différences finies", cmap="viridis", vmin=u_min, vmax=u_max)
    err = u_nn - u_fd
    e_max = np.percentile(np.abs(err), 99)   # échelle bornée au percentile 99 %
    im = ax[2].imshow(err.T, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
                      vmin=-e_max, vmax=e_max)
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04, extend="both")
    for xb in (J_A, J_B):
        ax[2].axvline(xb, color="tab:orange", ls="--", lw=1.2)
    ax[2].set_title(r"$u_{PINN} - u_{DF}$ (percentile 99 %)")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("y")
    j_mid = len(g) // 2
    ax[3].plot(g, u_fd[:, j_mid], "k", lw=2, label="DF")
    ax[3].plot(g, u_nn[:, j_mid], "tab:red", ls="--", lw=1.5, label="PINN")
    ax[3].axvspan(J_A, J_B, color="tab:orange", alpha=0.2)
    ax[3].set_title(f"coupe y = {g[j_mid]:.3f}")
    ax[3].set_xlabel("x")
    ax[3].legend(loc="best")
    ax[3].grid(alpha=0.3)
    fig.suptitle(f"Cycle {i}   ε = {EPS:.2e}   |   J_PINN/J_ref = {J_nn / J_REF:.4e}   "
                 f"J_DF/J_ref = {J_fd / J_REF:.4e}   (écart {abs(J_nn - J_fd) / J_fd:.2%})   |   "
                 f"erreur L2 relative sur u = {rel_l2:.3e}")
    fig.tight_layout()
    fname = f"{i}_{OUT_FIG_FD}"
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    return fname

def compare_fd(i):
    g = np.linspace(0.0, 1.0, N_FD + 2)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    pts = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1), device=DEVICE)
    with torch.no_grad():
        rho_grid = rho_model(pts).cpu().numpy().reshape(len(g), len(g))
        u_nn = u_model(pts).cpu().numpy().reshape(len(g), len(g))
        J_nn = float((WJ * u_model(XJ) ** 2).sum())
    u_fd = solve_fd(rho_grid, g)
    J_fd = float((_WJ_np.ravel() * RegularGridInterpolator((g, g), u_fd)(_XJ_np) ** 2).sum())
    rel_l2 = float(np.sqrt(((u_nn - u_fd) ** 2).mean() / (u_fd ** 2).mean()))
    fname = draw_fd_figure(i, g, u_nn, u_fd, J_nn, J_fd, rel_l2)
    return fname, J_nn, J_fd, rel_l2

# ----------------------------------------------------------------------------
# Entraînement : continuation sur EPS, chaque palier arrêté sur stabilité
# ----------------------------------------------------------------------------
print(f"Device : {DEVICE}   points de quadrature : {X.shape[0]} (+ {XJ.shape[0]} pour J)")
print(f"J_ref (rho homogène = V0) = {J_REF:.6e}")
for i in range(N_CYCLES):
    last_cycle = (i == N_CYCLES - 1)
    tol_rho = TOL_RHO_FINAL if last_cycle else TOL_RHO
    tol_res = TOL_RES_FINAL if last_cycle else TOL_RES
    print(f"\n===== Cycle {i}   {time.strftime('%H:%M:%S')}   EPS = {EPS:.3e}   (tol aire modifiée {tol_rho:.0e}, tol ||r|| {tol_res:.0e}) =====")
    t0 = time.time()
    n_adam = ADAM_ITERS if i == 0 else 0          # Adam au premier palier seulement
    opt_adam = torch.optim.Adam(params, lr=ADAM_LR)
    for it in range(1, n_adam + 1):
        opt_adam.zero_grad()
        total, parts = compute_losses()
        total.backward()
        opt_adam.step()
        record(parts)
        if it % LOG_EVERY == 0 or it == 1:
            log("Adam", it, parts)
    adam_ends.append(len(history["total"]))
    # Un seul L-BFGS par palier (historique conservé d'un tronçon à l'autre)
    opt_lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=LBFGS_CHUNK,
                                  max_eval=int(1.25 * LBFGS_CHUNK),
                                  history_size=50, tolerance_grad=1e-11,
                                  tolerance_change=1e-14,
                                  line_search_fn="strong_wolfe")
    n_eval = [0]
    def closure():
        opt_lbfgs.zero_grad()
        total, parts = compute_losses()
        total.backward()
        record(parts)
        n_eval[0] += 1
        if n_eval[0] % LOG_EVERY == 0:
            log("LBFGS eval", n_eval[0], parts)
        return total
    with torch.no_grad():
        rho_old = rho_model(X)
    loss_old = compute_losses()[1]["total"]
    status = f"plafond de {MAX_CHUNKS} tronçons atteint"
    for k in range(1, MAX_CHUNKS + 1):
        _ = opt_lbfgs.step(closure)
        with torch.no_grad():
            rho_new = rho_model(X)
        d_rho = (W * (rho_new - rho_old).abs()).sum().item()   # aire du design qui a changé
        rho_old = rho_new
        _, cur = compute_losses()
        res_l2 = np.sqrt(cur["residu"])
        d_loss = abs(cur["total"] - loss_old) / abs(loss_old)
        loss_old = cur["total"]
        print(f"  [tronçon {k:3d}] aire modifiée = {d_rho:.2e}   ||r||_L2 = {res_l2:.2e}   "
              f"variation relative loss = {d_loss:.2e}")
        if d_rho < tol_rho and res_l2 < tol_res:
            status = "stable (design et résidu sous tolérance)"
            break
        if d_loss < TOL_STAGNATION:
            status = "stagnation de la loss"
            if res_l2 >= tol_res:
                status += " avec ||r|| au-dessus de la tolérance : augmenter BETA"
            break
    print(f"Arrêt du palier : {status}")
    print(f"Temps du cycle : {time.time() - t0:.1f} s")
    _, final = compute_losses()
    log("Final", 0, final)
    print(f"J physique = {final['J/J_ref'] * J_REF:.6e}   "
          f"||r||_L2 = {np.sqrt(final['residu']):.3e}   "
          f"volume = {final['volume']:.5f} (cible {V0})")
    out_models = f"{i}_{OUT_MODELS}"
    torch.save({"u": u_model.state_dict(), "rho": rho_model.state_dict(), "eps": EPS},
               out_models)
    fig_name = plot_results(i)
    print(f"Figures : {fig_name}   Modèles : {out_models}")
    fig_fd, J_nn, J_fd, err_u = compare_fd(i)
    print(f"Comparaison DF : {fig_fd}   J_PINN/J_ref = {J_nn / J_REF:.4e}   "
          f"J_DF/J_ref = {J_fd / J_REF:.4e}   erreur L2 relative sur u = {err_u:.3e}")
    EPS *= EPS_FACTOR
