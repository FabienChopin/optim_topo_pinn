"""Quadrature de Gauss-Legendre composite sur un pavé [lo, hi]^d (produit tensoriel)."""

import itertools

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
        pts = np.array(list(itertools.product(x1, repeat=dim)))
        wts = np.prod(np.array(list(itertools.product(w1, repeat=dim))), axis=1)
        self.dim = dim
        self.element_edges = np.linspace(lo, hi, n_elements + 1)
        self.points = torch.tensor(pts, device=device).reshape(-1, dim)
        self.weights = torch.tensor(wts, device=device).reshape(-1, 1)

    def __len__(self) -> int:
        return self.points.shape[0]

    def integrate(self, values: torch.Tensor) -> torch.Tensor:
        return (self.weights * values).sum()
