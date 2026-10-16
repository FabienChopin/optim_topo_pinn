"""Problème : -div(A(rho) grad u) = f sur [0, 1]^d, u = 0 au bord, objectif J = int_bande u^2,
bande = [b0, b1] x [0, 1]^(d-1).

Loi matériau RAMP : A = a2 + (a1 - a2) rho / (1 + q (1 - rho)).
q = 0 : loi linéaire ; q = a1/a2 - 1 : loi harmonique (couches en série, mélange exact en 1D) ;
au-delà, le gris est pénalisé. q est modifié par palier par l'entraînement (self.q).

BandProblem fixe la physique (commune à toutes les dimensions). Chaque dimension fournit la
solution de référence pour rho donné sur une grille de cellules : exacte en 1D, volumes finis en 2D.
"""

import numpy as np
import torch

from .config import Config, ProblemConfig
from .exact1d import PiecewiseSolution
from .fv2d import FVSolution2D


class BandProblem:
    dim: int
    lo, hi = 0.0, 1.0

    def __init__(self, cfg: ProblemConfig):
        self.a1, self.a2, self.v0, self.f = cfg.a1, cfg.a2, cfg.v0, cfg.source
        self.band = tuple(cfg.band)
        self.q = 0.0

    # --- termes de l'EDP et de l'objectif, aux points de quadrature (N, d) ---
    def conductivity(self, rho):
        return self.a2 + (self.a1 - self.a2) * rho / (1.0 + self.q * (1.0 - rho))

    def dconductivity(self, rho):
        return (self.a1 - self.a2) * (1.0 + self.q) / (1.0 + self.q * (1.0 - rho)) ** 2

    def source(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[:, :1], self.f)

    def boundary_factor(self, x: torch.Tensor) -> torch.Tensor:
        return (x * (1.0 - x)).prod(dim=1, keepdim=True)

    def objective_density(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        lo, hi = self.band
        inside = (x[:, :1] >= lo) & (x[:, :1] <= hi)
        return inside * u ** 2


class BandProblem1D(BandProblem):
    """Référence exacte : A constant par morceaux, morceaux quelconques."""

    dim = 1

    def __init__(self, cfg: ProblemConfig):
        super().__init__(cfg)
        self.j_ref = float(self.exact_objective([0.0, 1.0], [self.v0]))   # rho = v0, loi linéaire (q = 0)

    def exact(self, edges, rho) -> PiecewiseSolution:
        return PiecewiseSolution(edges, self.conductivity(torch.as_tensor(rho, dtype=torch.float64)), self.f)

    def exact_objective(self, edges, rho) -> torch.Tensor:
        return self.exact(edges, rho).integral_u2(*self.band)

    def cells(self, n: int):
        """Grille uniforme de n cellules : bords et milieux (numpy)."""
        edges = np.linspace(self.lo, self.hi, n + 1)
        return edges, 0.5 * (edges[1:] + edges[:-1])


class BandProblem2D(BandProblem):
    """Référence par volumes finis sur n x n cellules (n multiple de 10 : bande alignée)."""

    dim = 2

    def __init__(self, cfg: ProblemConfig, n_grid: int):
        super().__init__(cfg)
        self.n_grid = n_grid
        self.j_ref = self.objective(np.full((n_grid, n_grid), self.v0))   # rho = v0, loi linéaire (q = 0)

    def solve(self, rho: np.ndarray) -> FVSolution2D:
        return FVSolution2D(self.conductivity(np.asarray(rho, dtype=float)), self.f)

    def objective(self, rho: np.ndarray) -> float:
        return self.solve(rho).integral_u2(*self.band)

    def cells(self, n: int):
        """Centres des n x n cellules : coordonnées 1D et points (n^2, 2), ordre [i, j] (i selon x)."""
        c = (np.arange(n) + 0.5) / n
        gx, gy = np.meshgrid(c, c, indexing="ij")
        return c, np.stack([gx.ravel(), gy.ravel()], axis=1)


def make_problem(cfg: Config) -> BandProblem:
    if cfg.problem.dim == 1:
        return BandProblem1D(cfg.problem)
    if cfg.problem.dim == 2:
        return BandProblem2D(cfg.problem, cfg.output.n_grid)
    raise ValueError(f"problem.dim = {cfg.problem.dim} : seuls 1 et 2 sont implémentés")
