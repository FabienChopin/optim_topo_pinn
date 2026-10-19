"""Quadrature de Gauss-Legendre composite sur un pavé [lo, hi]^d (produit tensoriel)."""

import numpy as np
import torch


def gauss_composite_1d(n_elements: int, n_gauss: int, lo: float = 0.0, hi: float = 1.0):
    """Points et poids (numpy) : n_gauss points de Gauss sur chacun des n_elements éléments égaux."""
    t, w = np.polynomial.legendre.leggauss(n_gauss)
    edges = np.linspace(lo, hi, n_elements + 1)
    half = 0.5 * np.diff(edges)[:, None]
    x = (0.5 * (edges[:-1, None] + edges[1:, None]) + half * t).ravel()
    return x, (half * w).ravel()


class Quadrature:
    """Points (N, d) et poids (N, 1) sur [lo, hi]^d ; int f ~ (weights * f(points)).sum()."""

    def __init__(self, dim: int, n_elements: int, n_gauss: int, lo: float = 0.0, hi: float = 1.0,
                 device: str = "cpu"):
        x1, w1 = gauss_composite_1d(n_elements, n_gauss, lo, hi)
        pts = np.stack([g.ravel() for g in np.meshgrid(*[x1] * dim, indexing="ij")], axis=1)
        wts = np.prod(np.stack([g.ravel() for g in np.meshgrid(*[w1] * dim, indexing="ij")], axis=1), axis=1)
        self.dim = dim
        self.n_elements, self.n_gauss = n_elements, n_gauss
        self.element_edges = np.linspace(lo, hi, n_elements + 1)
        dtype = torch.get_default_dtype()
        self.points = torch.tensor(pts, dtype=dtype, device=device).reshape(-1, dim)
        self.weights = torch.tensor(wts, dtype=dtype, device=device).reshape(-1, 1)
        # 1D : S_ij = int_{-1}^{t_i} l_j (l_j : polynôme de Lagrange des nœuds de Gauss t de l'élément de référence)
        t, w = np.polynomial.legendre.leggauss(n_gauss)
        coef = np.linalg.inv(np.polynomial.legendre.legvander(t, n_gauss - 1))
        self._cum = torch.tensor(np.polynomial.legendre.legval(t, np.polynomial.legendre.legint(coef, lbnd=-1)).T,
                                 dtype=dtype, device=device)
        self._ref_weights = torch.tensor(w, dtype=dtype, device=device)

    def cumulative(self, values: torch.Tensor) -> torch.Tensor:
        """1D : int_lo^x values aux points, exacte pour values polynomiale de degré < n_gauss par élément."""
        if self.dim != 1:
            raise ValueError("intégrale cumulée : 1D seulement")
        v = values.reshape(self.n_elements, self.n_gauss)
        half = 0.5 * (self.element_edges[1] - self.element_edges[0])
        within = half * v @ self._cum.T
        totals = half * v @ self._ref_weights
        before = torch.cumsum(totals, dim=0) - totals
        return (before[:, None] + within).reshape(-1, 1)

    def __len__(self) -> int:
        return self.points.shape[0]

    def integrate(self, values: torch.Tensor) -> torch.Tensor:
        return (self.weights * values).sum()


class PointSet:
    """Points (N, d) et poids (N, 1) quelconques, même interface que Quadrature pour la loss."""

    def __init__(self, points: torch.Tensor, weights: torch.Tensor):
        self.points, self.weights = points, weights

    def __len__(self) -> int:
        return self.points.shape[0]

    def integrate(self, values: torch.Tensor) -> torch.Tensor:
        return (self.weights * values).sum()

    @classmethod
    def monte_carlo(cls, dim: int, n: int, lo: float = 0.0, hi: float = 1.0, device: str = "cpu") -> "PointSet":
        """n points uniformes sur [lo, hi]^d, poids égaux : intégrale de Monte-Carlo, bruitée."""
        dtype = torch.get_default_dtype()
        points = lo + (hi - lo) * torch.rand(n, dim, dtype=dtype, device=device)
        return cls(points, torch.full((n, 1), (hi - lo) ** dim / n, dtype=dtype, device=device))
