"""Problème 1D : -(A(rho) u')' = f sur [0, 1], u = 0 au bord, objectif J = int_band u^2.

Loi matériau RAMP : A = a2 + (a1 - a2) rho / (1 + q (1 - rho)).
q = 0 : loi linéaire ; q = a1/a2 - 1 : loi harmonique (couches en série, mélange exact en 1D) ;
au-delà, le gris est pénalisé. q est modifié par palier par l'entraînement (self.q).

Ce qui dépend du problème (domaine, A, source, bord, objectif, solution exacte) est ici ;
un problème 2D ou 3D fournira les mêmes méthodes.
"""

import numpy as np
import torch

from .config import ProblemConfig
from .exact1d import PiecewiseSolution


class BandProblem1D:
    dim = 1
    lo, hi = 0.0, 1.0

    def __init__(self, cfg: ProblemConfig):
        self.a1, self.a2, self.v0, self.f = cfg.a1, cfg.a2, cfg.v0, cfg.source
        self.band = tuple(cfg.band)
        self.q = 0.0                                                       # J_ref en loi linéaire, fixe
        self.j_ref = float(self.exact_objective([0.0, 1.0], [self.v0]))   # rho = v0 partout

    # --- termes de l'EDP et de l'objectif, aux points de quadrature (N, d) ---
    def conductivity(self, rho):
        return self.a2 + (self.a1 - self.a2) * rho / (1.0 + self.q * (1.0 - rho))

    def source(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x[:, :1], self.f)

    def boundary_factor(self, x: torch.Tensor) -> torch.Tensor:
        return (x * (1.0 - x)).prod(dim=1, keepdim=True)

    def objective_density(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        lo, hi = self.band
        inside = (x[:, :1] >= lo) & (x[:, :1] <= hi)
        return inside * u ** 2

    # --- solution exacte pour rho constant par morceaux ---
    def exact(self, edges, rho) -> PiecewiseSolution:
        return PiecewiseSolution(edges, self.conductivity(torch.as_tensor(rho, dtype=torch.float64)), self.f)

    def exact_objective(self, edges, rho) -> torch.Tensor:
        return self.exact(edges, rho).integral_u2(*self.band)

    def cells(self, n: int):
        """Grille uniforme de n cellules : bords et milieux (numpy)."""
        edges = np.linspace(self.lo, self.hi, n + 1)
        return edges, 0.5 * (edges[1:] + edges[:-1])
