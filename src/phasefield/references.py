"""Références, calculées sans réseau.

1D (solution exacte) :
- binaire : rho in {0, 1}, K interfaces dont on optimise les positions (SLSQP, volume exact,
  plusieurs départs) ; on garde le meilleur K. C'est la cible.
- relaxé : rho in [0, 1] constant par cellule, gradient projeté sur {moyenne = v0}, pour la loi
  RAMP de paramètre q donné. Si son J est plus bas que le binaire, le gris est « moins cher »
  que tout design 0/1 avec cette loi.

2D (volumes finis, pas d'optimum garanti) : optimisation par densité classique, filtre de
densité + RAMP avec continuation de q + gradient adjoint + gradient projeté sur le volume.
- binaire : continuation jusqu'à q_penal (le gris est pénalisé), puis seuillage au volume V0 ;
  on garde le meilleur des designs seuillés.
- relaxé : continuation jusqu'à q_max (la loi finale du PINN), gris permis.
"""

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import minimize

from .config import Config, ReferenceConfig
from .problem import BandProblem, BandProblem1D, BandProblem2D


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


# ----------------------------------------------------------------------------
# 2D : optimisation par densité sur la grille volumes finis
# ----------------------------------------------------------------------------
def _density_filter(n: int, radius: float) -> sp.csr_matrix:
    """Filtre chapeau de rayon radius (en unités de longueur), lignes normalisées."""
    r = radius * n
    m = int(np.ceil(r))
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    rows, cols, vals = [], [], []
    for di in range(-m, m + 1):
        for dj in range(-m, m + 1):
            w = r - np.hypot(di, dj)
            if w <= 0:
                continue
            i2, j2 = i + di, j + dj
            ok = (i2 >= 0) & (i2 < n) & (j2 >= 0) & (j2 < n)
            rows.append((i * n + j)[ok])
            cols.append((i2 * n + j2)[ok])
            vals.append(np.full(ok.sum(), w))
    w = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n))
    return sp.diags(1.0 / np.asarray(w.sum(axis=1)).ravel()) @ w


def _project_volume_np(target: np.ndarray, v0: float) -> np.ndarray:
    lo, hi = -1.0, 1.0
    for _ in range(60):
        shift = 0.5 * (lo + hi)
        new = np.clip(target + shift, 0.0, 1.0)
        lo, hi = (lo, shift) if new.mean() > v0 else (shift, hi)
    return new


def density_optimization_2d(problem: BandProblem2D, rcfg: ReferenceConfig, q_final: float) -> np.ndarray:
    """Densité filtrée (n_cells x n_cells) après continuation de q de 0 à q_final."""
    n = rcfg.n_cells
    filt = _density_filter(n, rcfg.filter_radius)
    rho = np.full(n * n, problem.v0)
    q_saved = problem.q
    try:
        for q in (np.linspace(0.0, q_final, rcfg.n_q_steps) if q_final > 0 else [0.0]):
            problem.q = q
            move, j_old = 0.05, np.inf
            for _ in range(rcfg.iters_per_step):
                rho_f = (filt @ rho).reshape(n, n)
                sol = problem.solve(rho_f)
                j = sol.integral_u2(*problem.band)
                g = filt.T @ (sol.gradient_a(*problem.band) * problem.dconductivity(rho_f)).ravel()
                move = min(0.2, move * 1.05) if j < j_old else move * 0.5
                j_old = j
                rho = _project_volume_np(rho - move * g / np.abs(g).max(), problem.v0)
    finally:
        problem.q = q_saved
    return (filt @ rho).reshape(n, n)


def _to_grid(rho: np.ndarray, n_grid: int) -> np.ndarray:
    factor = n_grid // rho.shape[0]
    if factor * rho.shape[0] != n_grid:
        raise ValueError(f"output.n_grid = {n_grid} doit être un multiple de reference.n_cells = {rho.shape[0]}")
    return np.kron(rho, np.ones((factor, factor)))


def _threshold_at_volume(rho: np.ndarray, v0: float) -> np.ndarray:
    return (rho > np.quantile(rho, 1.0 - v0)).astype(float)


def references_2d(problem: BandProblem2D, cfg: Config) -> dict:
    rcfg, q_max = cfg.reference, cfg.continuation.q_max
    penal = density_optimization_2d(problem, rcfg, rcfg.q_penal)
    relaxed = density_optimization_2d(problem, rcfg, q_max)
    best = None
    for name, rho in (("pénalisé", penal), ("relaxé", relaxed)):
        binary = _to_grid(_threshold_at_volume(rho, problem.v0), problem.n_grid)
        j = problem.objective(binary) / problem.j_ref        # rho in {0, 1} : indépendant de q
        if best is None or j < best[0]:
            best = (j, binary, name)
    q_saved, problem.q = problem.q, q_max
    relaxed_grid = _to_grid(relaxed, problem.n_grid)
    j_relaxed = problem.objective(relaxed_grid) / problem.j_ref
    problem.q = q_saved
    j_b, binary, origin = best
    grey = float(np.mean(4.0 * relaxed * (1.0 - relaxed)))
    return {
        "binary": {"J": j_b, "rho_grid": binary,
                   "label": f"référence binaire (densité RAMP q -> {rcfg.q_penal:g}, seuillée au volume V0, "
                            f"design {origin}) : J/J_ref = {j_b:.4f}",
                   "info": {"J": j_b, "volume": float(binary.mean()), "origin": origin}},
        "relaxed": {"J": j_relaxed, "rho_grid": relaxed_grid, "grey": grey,
                    "label": f"référence relaxée (densité, gris permis, q = {q_max:g}) : "
                             f"J/J_ref = {j_relaxed:.4f}, gris {grey:.3f}",
                    "info": {"J": j_relaxed, "grey": grey, "q": q_max}},
    }


def references_1d(problem: BandProblem1D, cfg: Config) -> dict:
    b = binary_optimum(problem, seed=cfg.run.seed)
    r = relaxed_optimum(problem, q=cfg.continuation.q_max, binary=b)
    b["label"] = (f"optimum binaire : J/J_ref = {b['J']:.4f}, {b['n_interfaces']} interfaces en "
                  f"{np.round(b['edges'][1:-1], 4).tolist()}, ρ = {b['rho'].astype(int).tolist()}")
    b["info"] = {"J": b["J"], "interfaces": b["edges"][1:-1].tolist(), "rho": b["rho"].tolist()}
    r["label"] = (f"optimum relaxé (gris permis, q = {cfg.continuation.q_max:g}) : J/J_ref = {r['J']:.4f}, "
                  f"gris {r['grey']:.3f}")
    r["info"] = {"J": r["J"], "grey": r["grey"], "q": cfg.continuation.q_max}
    return {"binary": b, "relaxed": r}


def compute_references(problem: BandProblem, cfg: Config) -> dict:
    refs = references_1d(problem, cfg) if problem.dim == 1 else references_2d(problem, cfg)
    refs.update(band=problem.band, v0=problem.v0)
    return refs
