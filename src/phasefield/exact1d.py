"""Solution exacte en 1D de -(A u')' = f, u(0) = u(1) = 0, f constant, A constant par morceaux.

On intègre une fois : A u' = C - f x. Puis u(x) = int_0^x (C - f s) / A ds, avec C fixé par
u(1) = 0 : C = f int s / A / int 1 / A. Sur chaque morceau, u est un polynôme de degré 2 :
la solution et int u^2 (Gauss à 3 points par morceau) sont exactes.
Écrit en torch : dérivable par rapport aux valeurs de A (optimum relaxé).
"""

import numpy as np
import torch

_T3, _W3 = (torch.tensor(a) for a in np.polynomial.legendre.leggauss(3))


class PiecewiseSolution:
    def __init__(self, edges, a, source: float = 1.0):
        """edges (K+1,) croissants de 0 à 1, a (K,) conductivité de chaque morceau."""
        self.edges = torch.as_tensor(edges, dtype=torch.float64)
        self.a = torch.as_tensor(a, dtype=torch.float64)
        self.f = source
        length = self.edges[1:] - self.edges[:-1]
        half_sq = 0.5 * (self.edges[1:] ** 2 - self.edges[:-1] ** 2)
        self.flux0 = self.f * (half_sq / self.a).sum() / (length / self.a).sum()   # C
        increments = (self.flux0 * length - self.f * half_sq) / self.a
        self.u_edges = torch.cat([increments.new_zeros(1), torch.cumsum(increments, 0)])

    def _cell(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.searchsorted(self.edges, x, right=True) - 1
        return k.clamp(0, len(self.a) - 1)

    def u(self, x) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float64)
        k = self._cell(x)
        b = self.edges[k]
        return self.u_edges[k] + (self.flux0 * (x - b) - 0.5 * self.f * (x ** 2 - b ** 2)) / self.a[k]

    def du(self, x) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float64)
        return (self.flux0 - self.f * x) / self.a[self._cell(x)]

    def integral_u2(self, lo: float, hi: float) -> torch.Tensor:
        """int_lo^hi u^2 dx, exacte (morceaux coupés en lo et hi)."""
        inner = self.edges[(self.edges > lo) & (self.edges < hi)]
        cuts = torch.cat([inner.new_tensor([lo]), inner, inner.new_tensor([hi])])
        left, right = cuts[:-1], cuts[1:]
        mid, half = 0.5 * (left + right), 0.5 * (right - left)
        x = mid[:, None] + half[:, None] * _T3[None, :]
        return (half[:, None] * _W3[None, :] * self.u(x) ** 2).sum()
