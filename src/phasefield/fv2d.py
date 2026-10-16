"""Solveur de référence 2D : -div(A grad u) = f sur [0, 1]^2, u = 0 au bord, f constant.

Volumes finis à 5 points sur n x n cellules carrées (taille h = 1/n), A constant par cellule.
Conductance d'une face intérieure : moyenne harmonique 2 A_i A_j / (A_i + A_j) (flux continu
à travers un saut de matériau) ; face de bord : 2 A_i (bord à une demi-maille du centre).
Tableaux indexés [i, j] avec i selon x et j selon y ; inconnue k = i n + j.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class FVSolution2D:
    def __init__(self, a: np.ndarray, source: float = 1.0):
        self.a = np.asarray(a, dtype=float)
        self.n = n = self.a.shape[0]
        self.h = 1.0 / n
        a = self.a
        self.cx = 2.0 * a[:-1, :] * a[1:, :] / (a[:-1, :] + a[1:, :])   # faces (i, j) | (i + 1, j)
        self.cy = 2.0 * a[:, :-1] * a[:, 1:] / (a[:, :-1] + a[:, 1:])   # faces (i, j) | (i, j + 1)
        diag = np.zeros((n, n))
        diag[:-1, :] += self.cx
        diag[1:, :] += self.cx
        diag[:, :-1] += self.cy
        diag[:, 1:] += self.cy
        diag[0, :] += 2.0 * a[0, :]
        diag[-1, :] += 2.0 * a[-1, :]
        diag[:, 0] += 2.0 * a[:, 0]
        diag[:, -1] += 2.0 * a[:, -1]
        k = np.arange(n * n).reshape(n, n)
        rows = [k.ravel(), k[:-1, :].ravel(), k[1:, :].ravel(), k[:, :-1].ravel(), k[:, 1:].ravel()]
        cols = [k.ravel(), k[1:, :].ravel(), k[:-1, :].ravel(), k[:, 1:].ravel(), k[:, :-1].ravel()]
        vals = [diag.ravel(), -self.cx.ravel(), -self.cx.ravel(), -self.cy.ravel(), -self.cy.ravel()]
        matrix = sp.csc_matrix((np.concatenate(vals) / self.h ** 2, (np.concatenate(rows), np.concatenate(cols))),
                               shape=(n * n, n * n))
        self.lu = spla.splu(matrix)
        self.u = self.lu.solve(np.full(n * n, float(source))).reshape(n, n)

    def centers(self) -> np.ndarray:
        return (np.arange(self.n) + 0.5) * self.h

    def band_mask(self, lo: float, hi: float) -> np.ndarray:
        x = self.centers()
        return np.broadcast_to(((x >= lo) & (x <= hi))[:, None], (self.n, self.n)).astype(float)

    def integral_u2(self, lo: float, hi: float) -> float:
        """int_{[lo, hi] x [0, 1]} u^2, cellules dont le centre est dans la bande."""
        return float((self.band_mask(lo, hi) * self.u ** 2).sum() * self.h ** 2)

    def gradient_a(self, lo: float, hi: float) -> np.ndarray:
        """dJ/dA par cellule, par l'adjoint : K p = dJ/du, dJ/dA = -p^T (dK/dA) u (K symétrique)."""
        n, h, a, u = self.n, self.h, self.a, self.u
        p = self.lu.solve((2.0 * h ** 2 * self.band_mask(lo, hi) * u).ravel()).reshape(n, n)
        g = np.zeros((n, n))
        du, dp = u[:-1, :] - u[1:, :], p[:-1, :] - p[1:, :]          # faces en x
        s = a[:-1, :] + a[1:, :]
        g[:-1, :] += 2.0 * a[1:, :] ** 2 / s ** 2 * du * dp
        g[1:, :] += 2.0 * a[:-1, :] ** 2 / s ** 2 * du * dp
        du, dp = u[:, :-1] - u[:, 1:], p[:, :-1] - p[:, 1:]          # faces en y
        s = a[:, :-1] + a[:, 1:]
        g[:, :-1] += 2.0 * a[:, 1:] ** 2 / s ** 2 * du * dp
        g[:, 1:] += 2.0 * a[:, :-1] ** 2 / s ** 2 * du * dp
        for sl in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):   # faces de bord (coins : deux fois)
            g[sl] += 2.0 * u[sl] * p[sl]
        return -g / h ** 2
