"""Optimums de référence, calculés avec la solution exacte (sans réseau).

- binaire : rho in {0, 1}, K interfaces dont on optimise les positions (SLSQP, volume exact,
  plusieurs départs) ; on garde le meilleur K. C'est la cible.
- relaxé : rho in [0, 1] constant par cellule, gradient projeté sur {moyenne = v0}, pour la loi
  RAMP de paramètre q donné. Si son J est plus bas que le binaire, le gris est « moins cher »
  que tout design 0/1 avec cette loi.
"""

import numpy as np
import torch
from scipy.optimize import minimize

from .problem import BandProblem1D


def _binary_design(positions, first_phase: int):
    edges = np.concatenate([[0.0], np.sort(positions), [1.0]])
    rho = (first_phase + np.arange(len(edges) - 1)) % 2
    return edges, rho.astype(float)


def _simplify(edges, rho, tol: float = 1e-6):
    """Retire les morceaux de longueur nulle et fusionne les voisins de même phase."""
    keep = np.diff(edges) > tol
    rho = rho[keep]
    lengths = np.diff(edges)[keep]
    starts = np.r_[True, rho[1:] != rho[:-1]]
    rho = rho[starts]
    edges = np.r_[0.0, np.cumsum(lengths)[np.r_[starts[1:], True]]]
    edges[-1] = 1.0
    return edges, rho


def binary_optimum(problem: BandProblem1D, max_interfaces: int = 4, n_starts: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    best = None
    for k in range(1, max_interfaces + 1):
        for first in (0, 1):
            def objective(p, first=first):
                edges, rho = _binary_design(p, first)
                return float(problem.exact_objective(edges, rho)) / problem.j_ref

            def volume(p, first=first):
                edges, rho = _binary_design(p, first)
                return float(np.sum(np.diff(edges) * rho)) - problem.v0

            constraints = [{"type": "eq", "fun": volume}]
            if k > 1:
                constraints.append({"type": "ineq", "fun": lambda p: np.diff(p)})
            for _ in range(n_starts):
                p0 = np.sort(rng.uniform(0.0, 1.0, k))
                res = minimize(objective, p0, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                               constraints=constraints, options={"ftol": 1e-12, "maxiter": 500})
                if res.success and abs(volume(res.x)) < 1e-8 and (best is None or res.fun < best["J"]):
                    edges, rho = _simplify(*_binary_design(res.x, first))
                    best = {"J": float(res.fun), "edges": edges, "rho": rho, "n_interfaces": len(rho) - 1}
    return best


def relaxed_optimum(problem: BandProblem1D, q: float, binary: dict | None = None, n_cells: int = 500,
                    n_q_steps: int = 8, iters: int = 2000):
    """Pour q grand, le gris est presque isolant et la descente partie de rho = v0 reste bloquée :
    continuation en q (0 -> q, n_q_steps paliers), et second départ depuis l'optimum binaire s'il
    est donné (admissible, donc le résultat ne peut pas être pire). On garde le meilleur."""
    edges, mid = problem.cells(n_cells)
    edges_t = torch.tensor(edges)
    q_saved = problem.q
    try:
        rho = torch.full((n_cells,), problem.v0, dtype=torch.float64)
        for q_k in (np.linspace(0.0, q, n_q_steps)[:-1] if q > 0 else []):
            problem.q = q_k
            _, rho = _descend(problem, edges_t, rho, iters // 4)
        problem.q = q
        best = _descend(problem, edges_t, rho, iters)
        if binary is not None:
            rho_b = torch.tensor(binary["rho"][np.searchsorted(binary["edges"], mid, side="right") - 1])
            best = min(best, _descend(problem, edges_t, rho_b, iters), key=lambda t: t[0])
    finally:
        problem.q = q_saved
    j, rho = best
    rho = rho.numpy()
    return {"J": j, "edges": edges, "rho": rho, "grey": float(np.mean(4.0 * rho * (1.0 - rho)))}


def _project_volume(target: torch.Tensor, v0: float) -> torch.Tensor:
    lo, hi = -1.0, 1.0                     # bissection sur le décalage : moyenne = v0, rho in [0, 1]
    for _ in range(60):
        shift = 0.5 * (lo + hi)
        new = (target + shift).clamp(0.0, 1.0)
        lo, hi = (lo, shift) if new.mean() > v0 else (shift, hi)
    return new


def _descend(problem: BandProblem1D, edges_t: torch.Tensor, rho: torch.Tensor, iters: int):
    """Gradient projeté à pas adaptatif ; renvoie (meilleur J / J_ref, rho correspondant)."""
    rho = _project_volume(rho.clone(), problem.v0)
    move, j_old = 0.05, np.inf
    best = (np.inf, rho.clone())
    for _ in range(iters):
        r = rho.clone().requires_grad_(True)
        j = problem.exact_objective(edges_t, r) / problem.j_ref
        (g,) = torch.autograd.grad(j, r)
        j = j.item()
        if j < best[0]:
            best = (j, rho.clone())
        move = min(0.2, move * 1.05) if j < j_old else move * 0.5
        j_old = j
        rho = _project_volume(rho - move * g / g.abs().max(), problem.v0)
    return best
