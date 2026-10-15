# Discrétisation, réseaux, losses et solveur DF de référence.
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
from parametres import (DEVICE, A1, A2, V0, LAMBDA, MU, BETA, N_GAUSS, N_GAUSS_JX,
                        N_GAUSS_JY, J_A, J_B, WIDTH, DEPTH, N_FD)
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
XJ_np, WJ_np = gauss_legendre_2d(N_GAUSS_JX, N_GAUSS_JY, J_A, J_B, 0.0, 1.0)
X, W = torch.tensor(_X_np, device=DEVICE), torch.tensor(_W_np, device=DEVICE)
XJ, WJ = torch.tensor(XJ_np, device=DEVICE), torch.tensor(WJ_np, device=DEVICE)
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

# ----------------------------------------------------------------------------
# Opérateurs différentiels et losses
# ----------------------------------------------------------------------------
def grad(y, x):
    """Gradient de y (N, 1) par rapport à x (N, d) -> (N, d). Points indépendants."""
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

def divergence(q, x):
    """Divergence d'un champ q (N, d) -> (N, 1)."""
    return sum(grad(q[:, k:k + 1], x)[:, k:k + 1] for k in range(x.shape[1]))

def conductivity(rho, p):
    return A2 + (A1 - A2) * rho ** p

def compute_losses(u_model, rho_model, eps, p, j_ref):
    x = X.clone().requires_grad_(True)
    u = u_model(x)
    gu = grad(u, x)                                   # (N, 2)
    rho = rho_model(x)
    grho = grad(rho, x)                               # (N, 2)
    # Résidu fort : autograd sur le flux q = A(rho) grad u
    q = conductivity(rho, p) * gu
    r = -divergence(q, x) - 1.0
    L_res = (W * r ** 2).sum()                        # ||r||_{L2}^2
    # Van der Waals
    L_vdw = (W * (eps * (grho ** 2).sum(dim=1, keepdim=True)
                  + rho ** 2 * (1.0 - rho) ** 2 / eps)).sum()
    # Volume
    vol = (W * rho).sum()
    L_vol = (vol - V0) ** 2
    # Objectif sur la bande [J_A, J_B] x [0, 1] (jeu de points dédié)
    J = (WJ * u_model(XJ) ** 2).sum()
    total = J / j_ref + BETA * L_res + LAMBDA * L_vdw + MU * L_vol
    parts = {
        "total": total.item(),
        "J/J_ref": (J / j_ref).item(),
        "residu": L_res.item(),
        "vdw": L_vdw.item(),
        "vol_pen": L_vol.item(),
        "volume": vol.item(),
        "gris": (W * 4.0 * rho * (1.0 - rho)).sum().item(),   # 0 : noir et blanc, 1 : tout à 0.5
    }
    return total, parts

# ----------------------------------------------------------------------------
# Solveur DF de référence (rho figé)
# ----------------------------------------------------------------------------
def solve_fd(rho_grid, g, p):
    """-div(A(rho) grad u) = 1 sur [0,1]^2, u = 0 au bord. Volumes finis 5 points,
    moyenne harmonique de A aux faces. rho_grid : (n+2, n+2) sur la grille g, bords compris."""
    n = len(g) - 2
    h = g[1] - g[0]
    A = conductivity(rho_grid, p)
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

def j_fd(u_grid, g):
    """J d'un champ u donné sur la grille g (interpolé aux points de Gauss de la bande)."""
    return float((WJ_np.ravel() * RegularGridInterpolator((g, g), u_grid)(XJ_np) ** 2).sum())

G_FD = np.linspace(0.0, 1.0, N_FD + 2)

def j_homogene(p):
    """J du design homogène rho = V0 à l'exposant p, par DF (J_ref = j_homogene(1.0))."""
    return j_fd(solve_fd(np.full((len(G_FD), len(G_FD)), V0), G_FD, p), G_FD)
