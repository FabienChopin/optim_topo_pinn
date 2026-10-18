"""Évaluation du design courant avec la solution de référence (indépendante du PINN) :
exacte en 1D, volumes finis en 2D."""

import numpy as np
import torch


@torch.no_grad()
def eval_nets(u_net, rho_net, points: np.ndarray, batch: int = 65536):
    device = next(rho_net.parameters()).device
    points = points.reshape(len(points), -1)
    rho, u = [], []
    for k in range(0, len(points), batch):
        x = torch.tensor(points[k:k + batch], dtype=torch.get_default_dtype(), device=device)
        rho.append(rho_net(x).cpu().numpy().ravel())
        u.append(u_net(x).cpu().numpy().ravel())
    return np.concatenate(rho).astype(float), np.concatenate(u).astype(float)


def _l2_rel(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2) / np.mean(b ** 2)))


def interface_measure(binary: np.ndarray) -> float:
    """1D : nombre d'interfaces. 2D (grille n x n sur [0, 1]^2) : longueur d'interface, arêtes entre
    cellules de phases différentes corrigées de pi/4 (une courbe d'orientation quelconque a en
    moyenne 4/pi fois plus d'arêtes que sa longueur)."""
    if binary.ndim == 1:
        return int(np.count_nonzero(np.diff(binary)))
    edges = np.count_nonzero(np.diff(binary, axis=0)) + np.count_nonzero(np.diff(binary, axis=1))
    return 0.25 * np.pi * edges / binary.shape[0]


def evaluate(problem, u_net, rho_net, cfg_output, eps: float) -> dict:
    return (_evaluate_1d if problem.dim == 1 else _evaluate_2d)(problem, u_net, rho_net, cfg_output, eps)


def _evaluate_1d(problem, u_net, rho_net, cfg_output, eps):
    """rho du réseau échantillonné sur n_exact cellules, puis solution exacte pour ce rho."""
    edges, mid = problem.cells(cfg_output.n_exact)
    rho, u_pinn = eval_nets(u_net, rho_net, mid)
    exact = problem.exact(edges, rho)
    u_exact = exact.u(mid).numpy()
    binary = (rho > 0.5).astype(float)
    n_interfaces = interface_measure(binary)
    return {
        "x": mid, "rho": rho, "u_pinn": u_pinn, "u_exact": u_exact,
        "J_exact": float(exact.integral_u2(*problem.band)) / problem.j_ref,
        "J_binary": float(problem.exact_objective(edges, binary)) / problem.j_ref,
        "vol_binary": float(binary.mean()),
        "interfaces": n_interfaces, "interfaces_text": f"{n_interfaces} interfaces",
        "grey_floor": 4.0 * eps * n_interfaces,   # gris d'un profil sigmoïde(x / eps) par interface
        "u_error": _l2_rel(u_pinn, u_exact),
    }


def _evaluate_2d(problem, u_net, rho_net, cfg_output, eps):
    """rho du réseau aux centres de n_grid x n_grid cellules, puis volumes finis pour ce rho."""
    n = problem.n_grid
    c, pts = problem.cells(n)
    rho, u_pinn = (v.reshape(n, n) for v in eval_nets(u_net, rho_net, pts))
    sol = problem.solve(rho)
    binary = (rho > 0.5).astype(float)
    perimeter = interface_measure(binary)
    return {
        "x": c, "rho": rho, "u_pinn": u_pinn, "u_exact": sol.u,
        "J_exact": sol.integral_u2(*problem.band) / problem.j_ref,
        "J_binary": problem.objective(binary) / problem.j_ref,
        "vol_binary": float(binary.mean()),
        "interfaces": perimeter, "interfaces_text": f"périmètre {perimeter:.2f}",
        "grey_floor": 4.0 * eps * perimeter,
        "u_error": _l2_rel(u_pinn, sol.u),
    }
