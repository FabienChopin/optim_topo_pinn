"""Évaluation du design courant avec la solution exacte (indépendante du PINN)."""

import numpy as np
import torch


@torch.no_grad()
def evaluate(problem, u_net, rho_net, n_cells: int, eps: float) -> dict:
    """rho du réseau échantillonné sur n_cells cellules, puis solution exacte pour ce rho."""
    device = next(rho_net.parameters()).device
    edges, mid = problem.cells(n_cells)
    x = torch.tensor(mid, device=device).reshape(-1, 1)
    rho = rho_net(x).cpu().numpy().ravel()
    u_pinn = u_net(x).cpu().numpy().ravel()
    exact = problem.exact(edges, rho)
    u_exact = exact.u(mid).numpy()
    binary = (rho > 0.5).astype(float)
    n_interfaces = int(np.count_nonzero(np.diff(binary)))
    return {
        "x": mid, "rho": rho, "u_pinn": u_pinn, "u_exact": u_exact,
        "J_exact": float(exact.integral_u2(*problem.band)) / problem.j_ref,
        "J_binary": float(problem.exact_objective(edges, binary)) / problem.j_ref,
        "vol_binary": float(binary.mean()),
        "n_interfaces": n_interfaces,
        "grey_floor": 4.0 * eps * n_interfaces,   # gris d'un profil sigmoïde(x / eps) par interface
        "u_error": float(np.sqrt(np.mean((u_pinn - u_exact) ** 2) / np.mean(u_exact ** 2))),
    }
