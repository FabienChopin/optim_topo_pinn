"""Optimisation topologique par champ de phase et PINN, en JAX, en 1D ou 2D (voir README.md).

    min J = ∫_bande u²,  bande = [b0, b1] x [0, 1]^(d-1)
    s.c. -div(A(ρ) ∇u) = f sur [0, 1]^d, u = 0 au bord, ∫ρ = V0

Exemples :
    python phasefield.py                                   # 1D, Adam puis L-BFGS à chaque palier en ε
    python phasefield.py --dim 2                           # 2D (GPU conseillé)
    python phasefield.py --opt lbfgs:1000 ng:20            # gradient naturel en fin de chaque palier
    python phasefield.py --q-max 99 --q-start 4            # loi harmonique à partir du palier 5

Pendant le calcul, runs/latest/live.npz contient l'état courant ; plotphase.py en tire les figures, ici ou
ailleurs. À la fin : runs/<date>_<nom>/ contient live.npz, log.txt, metrics.json, config.json, params/.
"""

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.sparse as sp
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
from scipy.sparse.linalg import splu

jax.config.update("jax_enable_x64", True)
try:   # figures sur place si matplotlib est installé ; sinon, python plotphase.py ailleurs
    import plotphase
    plotphase.use_files()
except ImportError:
    plotphase = None

EVAL_BATCH = 8192     # points par paquet pour évaluer les réseaux sur une grille
T3, W3 = np.polynomial.legendre.leggauss(3)
# valeurs par défaut qui dépendent de la dimension : (1D, 2D)
DIM_DEFAULTS = {"elements": (100, 100), "gauss": (10, 3), "width": (20, 32), "eps_min": (4e-3, 8e-3),
                "n_stages": (10, 8), "plot_every": (100, 200)}


# --- solutions de référence ---
class Exact1D:
    """-(A u')' = f, u(0) = u(1) = 0, A constant par morceaux sur edges. A u' = C - f x : u est de degré 2 par
    morceau, donc u et ∫_bande u² (Gauss à 3 points par morceau) sont exacts. a peut être un tableau jnp."""

    def __init__(self, edges, f, band):
        self.edges, self.f = np.asarray(edges, dtype=float), f
        lo, hi = band
        inner = self.edges[(self.edges > lo) & (self.edges < hi)]
        cuts = np.r_[lo, inner, hi]
        mid, half = 0.5 * (cuts[1:] + cuts[:-1]), 0.5 * np.diff(cuts)
        self.qx, self.qw = (mid[:, None] + half[:, None] * T3).ravel(), (half[:, None] * W3).ravel()
        self.length = np.diff(self.edges)
        self.half_sq = 0.5 * (self.edges[1:] ** 2 - self.edges[:-1] ** 2)

    def piece(self, x):
        return np.clip(np.searchsorted(self.edges, x, side="right") - 1, 0, len(self.length) - 1)

    def u(self, a, x, xp=np):
        c = self.f * xp.sum(self.half_sq / a) / xp.sum(self.length / a)
        u_edges = xp.concatenate([xp.zeros(1), xp.cumsum((c * self.length - self.f * self.half_sq) / a)])
        k = self.piece(x)
        b = self.edges[k]
        return u_edges[k] + (c * (x - b) - 0.5 * self.f * (x ** 2 - b ** 2)) / a[k]

    def band_u2(self, a, xp=np):
        return xp.sum(self.qw * self.u(a, self.qx, xp) ** 2)


class FV2D:
    """-div(A ∇u) = f sur [0, 1]², u = 0 au bord : volumes finis à 5 points sur n x n cellules, A constant par
    cellule. Faces intérieures : moyenne harmonique de A ; faces de bord : 2 A (bord à une demi-maille).
    Tableaux [i, j], i selon x ; inconnue k = i n + j."""

    def __init__(self, a, f):
        self.a = a = np.asarray(a, dtype=float)
        self.n = n = a.shape[0]
        self.h = 1.0 / n
        self.cx = 2.0 * a[:-1, :] * a[1:, :] / (a[:-1, :] + a[1:, :])
        self.cy = 2.0 * a[:, :-1] * a[:, 1:] / (a[:, :-1] + a[:, 1:])
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
        self.lu = splu(matrix)
        self.u = self.lu.solve(np.full(n * n, float(f))).reshape(n, n)

    def band_mask(self, band):
        x = (np.arange(self.n) + 0.5) * self.h
        return np.broadcast_to(((x >= band[0]) & (x <= band[1]))[:, None], (self.n, self.n)).astype(float)

    def band_u2(self, band):
        """∫_bande u², cellules dont le centre est dans la bande."""
        return float((self.band_mask(band) * self.u ** 2).sum() * self.h ** 2)

    def gradient_a(self, band):
        """dJ/dA par cellule, par l'adjoint : K p = dJ/du, dJ/dA = -pᵀ (dK/dA) u (K symétrique)."""
        n, h, a, u = self.n, self.h, self.a, self.u
        p = self.lu.solve((2.0 * h ** 2 * self.band_mask(band) * u).ravel()).reshape(n, n)
        g = np.zeros((n, n))
        du, dp = u[:-1, :] - u[1:, :], p[:-1, :] - p[1:, :]
        s = a[:-1, :] + a[1:, :]
        g[:-1, :] += 2.0 * a[1:, :] ** 2 / s ** 2 * du * dp
        g[1:, :] += 2.0 * a[:-1, :] ** 2 / s ** 2 * du * dp
        du, dp = u[:, :-1] - u[:, 1:], p[:, :-1] - p[:, 1:]
        s = a[:, :-1] + a[:, 1:]
        g[:, :-1] += 2.0 * a[:, 1:] ** 2 / s ** 2 * du * dp
        g[:, 1:] += 2.0 * a[:, :-1] ** 2 / s ** 2 * du * dp
        for sl in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):   # faces de bord (coins : deux fois)
            g[sl] += 2.0 * u[sl] * p[sl]
        return -g / h ** 2


# --- problème ---
class Problem:
    """Physique commune ; la référence (diagnostics, J_ref) est exacte en 1D, volumes finis en 2D.
    Loi RAMP : A = a2 + (a1 - a2) ρ / (1 + q (1 - ρ)) ; q = 0 linéaire, q = a1/a2 - 1 harmonique."""

    def __init__(self, a):
        self.dim, self.a1, self.a2, self.v0, self.f = a.dim, a.a1, a.a2, a.v0, a.source
        self.band = tuple(a.band)
        if self.dim == 1:
            self.n = a.n_exact
            self.edges = np.linspace(0.0, 1.0, self.n + 1)
            self.mid = 0.5 * (self.edges[1:] + self.edges[:-1])
            self.cells = self.mid[:, None]
            self.exact = Exact1D(self.edges, self.f, self.band)
            a_ref = np.array([self.conductivity(self.v0, 0.0)])   # ρ = V0 partout, loi linéaire
            self.j_ref = float(Exact1D([0.0, 1.0], self.f, self.band).band_u2(a_ref))
        else:
            self.n = a.n_grid
            self.mid = (np.arange(self.n) + 0.5) / self.n
            gx, gy = np.meshgrid(self.mid, self.mid, indexing="ij")
            self.cells = np.stack([gx.ravel(), gy.ravel()], 1)
            self.j_ref = self.objective_2d(np.full((self.n, self.n), self.v0), 0.0)

    def conductivity(self, rho, q):
        return self.a2 + (self.a1 - self.a2) * rho / (1.0 + q * (1.0 - rho))

    def dconductivity(self, rho, q):
        return (self.a1 - self.a2) * (1.0 + q) / (1.0 + q * (1.0 - rho)) ** 2

    def solve_2d(self, rho, q):
        return FV2D(self.conductivity(np.asarray(rho, dtype=float), q), self.f)

    def objective_2d(self, rho, q):
        return self.solve_2d(rho, q).band_u2(self.band)


def stage_eps(a, s):
    return a.eps0 if a.n_stages == 1 else a.eps0 * (a.eps_min / a.eps0) ** (s / (a.n_stages - 1))


def stage_q(a, s):
    """q0 jusqu'au palier q_start inclus, puis linéaire jusqu'à q_max au dernier palier."""
    last = a.n_stages - 1
    if s <= a.q_start or last <= a.q_start:
        return a.q0
    return a.q0 + (a.q_max - a.q0) * (s - a.q_start) / (last - a.q_start)


# --- références sans réseau ---
def _binary_design(positions, first):
    edges = np.concatenate([[0.0], np.sort(positions), [1.0]])
    return edges, ((first + np.arange(len(edges) - 1)) % 2).astype(float)


def _simplify(edges, rho, tol=1e-6):
    """Retire les morceaux de longueur nulle et fusionne les voisins de même phase."""
    keep = np.diff(edges) > tol
    rho, lengths = rho[keep], np.diff(edges)[keep]
    starts = np.r_[True, rho[1:] != rho[:-1]]
    rho = rho[starts]
    edges = np.r_[0.0, np.cumsum(lengths)[np.r_[starts[1:], True]]]
    edges[-1] = 1.0
    return edges, rho


def binary_optimum_1d(pb: Problem, seed, max_interfaces=4, n_starts=10):
    """ρ ∈ {0, 1}, K interfaces aux positions optimisées (SLSQP, volume exact, plusieurs départs)."""
    rng = np.random.default_rng(seed)
    best = None
    for k in range(1, max_interfaces + 1):
        for first in (0, 1):
            def objective(p, first=first):
                edges, rho = _binary_design(p, first)
                return float(Exact1D(edges, pb.f, pb.band).band_u2(pb.conductivity(rho, 0.0))) / pb.j_ref

            def volume(p, first=first):
                edges, rho = _binary_design(p, first)
                return float(np.sum(np.diff(edges) * rho)) - pb.v0

            constraints = [{"type": "eq", "fun": volume}]
            if k > 1:
                constraints.append({"type": "ineq", "fun": lambda p: np.diff(p)})
            for _ in range(n_starts):
                res = minimize(objective, np.sort(rng.uniform(0.0, 1.0, k)), method="SLSQP", bounds=[(0.0, 1.0)] * k,
                               constraints=constraints, options={"ftol": 1e-12, "maxiter": 500})
                if res.success and abs(volume(res.x)) < 1e-8 and (best is None or res.fun < best["J"]):
                    edges, rho = _simplify(*_binary_design(res.x, first))
                    best = {"J": float(res.fun), "edges": edges, "rho": rho, "n_interfaces": len(rho) - 1}
    return best


def _project_volume(target, v0):
    lo, hi = -1.0, 1.0      # bissection sur le décalage : moyenne = v0, ρ ∈ [0, 1]
    for _ in range(60):
        shift = 0.5 * (lo + hi)
        new = np.clip(target + shift, 0.0, 1.0)
        lo, hi = (lo, shift) if new.mean() > v0 else (shift, hi)
    return new


def relaxed_optimum_1d(pb: Problem, q, binary=None, n_cells=500, n_q_steps=8, iters=2000):
    """ρ ∈ [0, 1] par cellule, gradient projeté sur {moyenne = V0}. Pour q grand, le gris est presque isolant et
    la descente partie de ρ = V0 reste bloquée : continuation en q, et second départ depuis l'optimum binaire."""
    edges = np.linspace(0.0, 1.0, n_cells + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    ex = Exact1D(edges, pb.f, pb.band)
    value_grad = jax.jit(jax.value_and_grad(lambda r, q: ex.band_u2(pb.conductivity(r, q), jnp) / pb.j_ref))

    def descend(rho, q, iters):
        rho = _project_volume(rho, pb.v0)
        move, j_old, best = 0.05, np.inf, (np.inf, rho)
        for _ in range(iters):
            j, g = value_grad(jnp.asarray(rho), q)
            j, g = float(j), np.asarray(g)
            if j < best[0]:
                best = (j, rho)
            move = min(0.2, move * 1.05) if j < j_old else move * 0.5
            j_old = j
            rho = _project_volume(rho - move * g / np.abs(g).max(), pb.v0)
        return best

    rho = np.full(n_cells, pb.v0)
    for q_k in (np.linspace(0.0, q, n_q_steps)[:-1] if q > 0 else []):
        rho = descend(rho, q_k, iters // 4)[1]
    best = descend(rho, q, iters)
    if binary is not None:
        rho_b = binary["rho"][np.searchsorted(binary["edges"], mid, side="right") - 1]
        best = min(best, descend(rho_b, q, iters), key=lambda t: t[0])
    j, rho = best
    return {"J": j, "x": mid, "rho": rho, "grey": float(np.mean(4.0 * rho * (1.0 - rho)))}


def _density_filter(n, radius):
    """Filtre chapeau de rayon radius (unités de longueur), lignes normalisées."""
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


def density_optimization_2d(pb: Problem, a, q_final):
    """Densité filtrée (ref_cells²) après continuation de q de 0 à q_final. À q fixé : L-BFGS-B sur ρ ∈ [0, 1],
    volume par lagrangien augmenté (J / J_ref + m (V - V0) + c/2 (V - V0)², m mis à jour 4 fois)."""
    n = a.ref_cells
    filt = _density_filter(n, a.ref_filter)
    rho = np.full(n * n, pb.v0)
    multiplier, penalty, n_outer = 0.0, 100.0, 4

    def fun(r, q):
        rho_f = (filt @ r).reshape(n, n)
        sol = pb.solve_2d(rho_f, q)
        dv = r.mean() - pb.v0
        value = sol.band_u2(pb.band) / pb.j_ref + multiplier * dv + 0.5 * penalty * dv ** 2
        grad = filt.T @ (sol.gradient_a(pb.band) * pb.dconductivity(rho_f, q)).ravel() / pb.j_ref
        return value, grad + (multiplier + penalty * dv) / r.size

    for q in np.linspace(0.0, q_final, a.ref_q_steps):
        for _ in range(n_outer):
            rho = minimize(fun, rho, args=(q,), jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * rho.size,
                           options={"maxiter": max(1, a.ref_iters // n_outer)}).x
            multiplier += penalty * (rho.mean() - pb.v0)
    return (filt @ _project_volume(rho, pb.v0)).reshape(n, n)


def references(pb: Problem, a, log):
    """Optimum binaire (la cible) et optimum relaxé (gris permis, loi finale q_max)."""
    t0 = time.time()
    if pb.dim == 1:
        b = binary_optimum_1d(pb, a.seed)
        r = relaxed_optimum_1d(pb, a.q_max, binary=b)
        b["label"] = (f"optimum binaire : J/J_ref = {b['J']:.4f}, {b['n_interfaces']} interfaces en "
                      f"{np.round(b['edges'][1:-1], 4).tolist()}, ρ = {b['rho'].astype(int).tolist()}")
        r["label"] = f"optimum relaxé (gris permis, q = {a.q_max:g}) : J/J_ref = {r['J']:.4f}, gris {r['grey']:.3f}"
    else:
        factor = pb.n // a.ref_cells
        if factor * a.ref_cells != pb.n:
            raise SystemExit(f"--n-grid {pb.n} doit être un multiple de --ref-cells {a.ref_cells}")
        up = lambda rho: np.kron(rho, np.ones((factor, factor)))  # noqa: E731
        best = None
        relaxed = None
        for name, q in (("pénalisé", a.ref_q_penal), ("relaxé", a.q_max)):
            rho = density_optimization_2d(pb, a, q)
            relaxed = rho
            binary = up((rho > np.quantile(rho, 1.0 - pb.v0)).astype(float))   # seuillé au volume V0
            j = pb.objective_2d(binary, 0.0) / pb.j_ref                          # ρ ∈ {0, 1} : indépendant de q
            if best is None or j < best["J"]:
                best = {"J": j, "rho": binary, "origin": name}
        b = best
        b["label"] = (f"référence binaire (densité RAMP q -> {a.ref_q_penal:g}, seuillée au volume V0, design "
                      f"{b['origin']}) : J/J_ref = {b['J']:.4f}")
        rho_r = up(relaxed)
        r = {"J": pb.objective_2d(rho_r, a.q_max) / pb.j_ref, "rho": rho_r,
             "grey": float(np.mean(4.0 * relaxed * (1.0 - relaxed)))}
        r["label"] = (f"référence relaxée (densité, gris permis, q = {a.q_max:g}) : J/J_ref = {r['J']:.4f}, "
                      f"gris {r['grey']:.3f}")
    log(b["label"])
    log(r["label"])
    log(f"références calculées en {time.time() - t0:.1f} s")
    return {"binary": b, "relaxed": r}


# --- quadrature ---
def gauss_composite(n_el, n_g):
    t, w = np.polynomial.legendre.leggauss(n_g)
    edges = np.linspace(0.0, 1.0, n_el + 1)
    half = 0.5 * np.diff(edges)[:, None]
    return (0.5 * (edges[:-1, None] + edges[1:, None]) + half * t).ravel(), (half * w).ravel()


def quadrature(dim, n_el, n_g):
    """Gauss composite tensorisé sur [0, 1]^dim : points (N, dim), poids (N,)."""
    x, w = gauss_composite(n_el, n_g)
    pts = np.stack([g.ravel() for g in np.meshgrid(*[x] * dim, indexing="ij")], 1)
    wts = np.prod(np.stack([g.ravel() for g in np.meshgrid(*[w] * dim, indexing="ij")], 1), axis=1)
    return pts, wts


def batches(chunk, **arrays):
    """Paquets de chunk points (axe 0), le dernier complété par des poids nuls (sw = 0 annule le résidu)."""
    n = len(arrays["sw"])
    m = -(-n // chunk) * chunk
    out = {}
    for k, v in arrays.items():
        pad = np.full((m - n,) + v.shape[1:], 0.5 if k == "x" else 0.0)
        out[k] = jnp.asarray(np.concatenate([v, pad]).reshape((m // chunk, chunk) + v.shape[1:]))
    return out


# --- réseaux ---
def init_mlp(key, dim, width, depth):
    sizes = [dim] + [width] * depth + [1]
    params = []
    for k, (m, n) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:], strict=True), strict=True):
        params.append((jax.random.normal(k, (n, m)) * np.sqrt(2.0 / (m + n)), jnp.zeros(n)))
    return params


def mlp(p, x):
    h = 2.0 * x - 1.0
    for w, b in p[:-1]:
        h = jnp.tanh(w @ h + b)
    w, b = p[-1]
    return (w @ h + b)[0]


def u_fn(pu, x):
    return jnp.prod(x * (1.0 - x)) * mlp(pu, x)


def rho_fn(pr, x):
    return jax.nn.sigmoid(mlp(pr, x))


def init_rho(key, dim, width, depth, value):
    p = init_mlp(key, dim, width, depth)
    w, _ = p[-1]
    p[-1] = (0.1 * w, jnp.full(1, np.log(value / (1.0 - value))))
    return p


@jax.jit
def _on_grid(p, pts):
    return jax.lax.map(lambda x: mlp(p, x), pts, batch_size=EVAL_BATCH)


def rho_grid(pr, pts):
    return np.asarray(jax.nn.sigmoid(_on_grid(pr, jnp.asarray(pts))), dtype=float)


def u_grid(pu, pts):
    return np.prod(pts * (1.0 - pts), axis=1) * np.asarray(_on_grid(pu, jnp.asarray(pts)), dtype=float)


# --- loss de moindres carrés ---
class Loss:
    """L = J/J_ref + β ∫ (-div(A(ρ) ∇u) - f)² + λ ∫ (ε |∇ρ|² + ρ² (1 - ρ)² / ε) + μ (∫ρ - V0)².
    hp = (ε, q) varie d'un palier à l'autre sans recompilation."""

    TERMS = ("J", "residual", "vdw", "volume")

    def __init__(self, pb: Problem, a, unravel):
        x, w = quadrature(pb.dim, a.elements, a.gauss)
        inside = ((x[:, 0] >= pb.band[0]) & (x[:, 0] <= pb.band[1])).astype(float)
        data = {"x": x, "sw": np.sqrt(w), "band": inside}
        self.data = {k: jnp.asarray(v) for k, v in data.items()}
        self.batches = batches(a.chunk, **data)
        self.w = jnp.asarray(w)
        self.pb, self.a, self.unravel = pb, a, unravel

    def fields(self, theta, x, q):
        pb = self.pb
        pu, pr = self.unravel(theta)

        def point(x):
            rho, drho = jax.value_and_grad(rho_fn, 1)(pr, x)
            flux = lambda y: pb.conductivity(rho_fn(pr, y), q) * jax.grad(u_fn, 1)(pu, y)  # noqa: E731
            return u_fn(pu, x), rho, drho, jnp.trace(jax.jacfwd(flux)(x))

        return jax.vmap(point)(x)

    def residuals(self, fields, d, hp):
        """Résidus ponctuels (points, 3 + dim) : J, résidu de l'EDP, Van der Waals (∇ρ, double puits)."""
        pb, a = self.pb, self.a
        u, rho, drho, div = fields
        eps, sw = hp[0], d["sw"]
        return jnp.concatenate([
            (sw * d["band"] * u / np.sqrt(pb.j_ref))[:, None],
            (np.sqrt(a.beta) * sw * (-div - pb.f))[:, None],
            jnp.sqrt(a.lam * eps) * sw[:, None] * drho,
            (jnp.sqrt(a.lam / eps) * sw * rho * (1.0 - rho))[:, None],
        ], 1)

    def local(self, theta, d, hp):
        return self.residuals(self.fields(theta, d["x"], hp[1]), d, hp)

    def glob(self, theta, hp):
        """Résidu global : volume."""
        rho = jax.vmap(rho_fn, (None, 0))(self.unravel(theta)[1], self.data["x"])
        return np.sqrt(self.a.mu) * jnp.atleast_1d(jnp.sum(self.w * rho) - self.pb.v0)

    def stats(self, theta, hp):
        """L, et (J, résidu, Van der Waals, volume pondérés, ∫ρ, gris ∫4ρ(1-ρ)), d'un seul calcul des champs."""
        f = self.fields(theta, self.data["x"], hp[1])
        r2 = self.residuals(f, self.data, hp) ** 2
        rho = f[1]
        vol = jnp.sum(self.w * rho)
        terms = jnp.stack([r2[:, 0].sum(), r2[:, 1].sum(), r2[:, 2:].sum(), self.a.mu * (vol - self.pb.v0) ** 2])
        return terms.sum(), jnp.concatenate([terms, jnp.stack([vol, jnp.sum(self.w * 4.0 * rho * (1.0 - rho))])])

    def value(self, theta, hp):
        return self.stats(theta, hp)[0]


# --- optimiseurs ---
def adam_runner(loss: Loss, lr):
    opt = optax.adam(lr)

    @partial(jax.jit, static_argnums=2)
    def chunk(theta, state, n, hp):
        def body(carry, _):
            th, st = carry
            (_, stats), g = jax.value_and_grad(loss.stats, has_aux=True)(th, hp)
            up, st = opt.update(g, st)
            return (th + up, st), stats
        (theta, state), stats = jax.lax.scan(body, (theta, state), None, length=n)
        return theta, state, stats
    return opt.init, chunk


def lbfgs_runner(loss: Loss, memory):
    opt = optax.lbfgs(memory_size=memory)

    @partial(jax.jit, static_argnums=2)
    def chunk(theta, state, n, hp):
        value = partial(loss.value, hp=hp)
        vg = optax.value_and_grad_from_state(value)

        def body(carry, _):
            th, st = carry
            v, g = vg(th, state=st)
            up, st = opt.update(g, st, th, value=v, grad=g, value_fn=value)
            th = optax.apply_updates(th, up)
            return (th, st), loss.stats(th, hp)[1]
        (theta, state), stats = jax.lax.scan(body, (theta, state), None, length=n)
        return theta, state, stats
    return opt.init, chunk


class NaturalGradient:
    """Gradient naturel de la loss de moindres carrés L = |r(θ)|², c'est-à-dire Gauss-Newton amorti
    (Levenberg-Marquardt) : (JᵀJ + μ max diag(JᵀJ) I) d = Jᵀr, θ ← θ - d.
    Gain = baisse réelle / baisse prédite |r|² - |r - J d|² : refus et μ × 10 si < 1/4, μ / 10 si > 3/4.
    JᵀJ (taille θ²) est accumulé par paquets de points : la Jacobienne complète n'est jamais formée."""

    def __init__(self, loss: Loss, damping=1e-8, min_damping=1e-12, max_tries=15):
        self.damping, self.min_damping, self.max_tries = damping, min_damping, max_tries
        flat = lambda th, b, hp: loss.local(th, b, hp).ravel()  # noqa: E731

        def total(theta, hp):
            def body(acc, b):
                r = flat(theta, b, hp)
                return acc + r @ r, None
            sq, _ = jax.lax.scan(body, 0.0, loss.batches)
            return sq + jnp.sum(loss.glob(theta, hp) ** 2)

        @jax.jit
        def prepare(theta, hp):
            p = theta.size

            def body(carry, b):
                gram, g, sq = carry
                r = flat(theta, b, hp)
                jac = (jax.jacrev if r.size < p else jax.jacfwd)(flat)(theta, b, hp)
                return (gram + jac.T @ jac, g + jac.T @ r, sq + r @ r), None
            (gram, g, sq), _ = jax.lax.scan(body, (jnp.zeros((p, p)), jnp.zeros(p), 0.0), loss.batches)
            rg, jg = loss.glob(theta, hp), jax.jacrev(loss.glob)(theta, hp)
            return sq + rg @ rg, gram + jg.T @ jg, g + jg.T @ rg

        @jax.jit
        def trial(theta, gram, g, mu, hp):
            scale = mu * jnp.max(jnp.diag(gram))
            chol = jnp.linalg.cholesky(gram + scale * jnp.eye(len(g)))
            d = jax.scipy.linalg.cho_solve((chol, True), g)
            predicted = 2.0 * g @ d - d @ gram @ d
            new = theta - d
            return new, total(new, hp), predicted

        self.prepare, self.trial = prepare, trial

    def step(self, theta, hp):
        """Un pas ; rend (θ, accepté)."""
        l0, gram, g = self.prepare(theta, hp)
        for _ in range(self.max_tries):
            new, l1, predicted = self.trial(theta, gram, g, self.damping, hp)
            gain = float((l0 - l1) / predicted)
            if np.isfinite(gain) and gain > 0.25:
                if gain > 0.75:
                    self.damping = max(self.damping / 10.0, self.min_damping)
                return new, True
            self.damping *= 10.0
        return theta, False


# --- diagnostics ---
def evaluate(pb: Problem, theta, unravel, eps, q):
    """Design courant évalué par la référence (exacte en 1D, volumes finis en 2D), indépendante du PINN."""
    pu, pr = unravel(theta)
    rho, u_pinn = rho_grid(pr, pb.cells), u_grid(pu, pb.cells)
    binary = (rho > 0.5).astype(float)
    if pb.dim == 1:
        a = pb.conductivity(rho, q)
        u_exact = pb.exact.u(a, pb.mid)
        j_exact = float(pb.exact.band_u2(a)) / pb.j_ref
        j_binary = float(pb.exact.band_u2(pb.conductivity(binary, 0.0))) / pb.j_ref
        interfaces = int(np.count_nonzero(np.diff(binary)))
        text = f"{interfaces} interfaces"
    else:
        rho, u_pinn, binary = (v.reshape(pb.n, pb.n) for v in (rho, u_pinn, binary))
        sol = pb.solve_2d(rho, q)
        u_exact = sol.u
        j_exact = sol.band_u2(pb.band) / pb.j_ref
        j_binary = pb.objective_2d(binary, 0.0) / pb.j_ref
        # longueur d'interface : arêtes entre cellules de phases différentes, corrigées de π/4
        # (une courbe d'orientation quelconque a en moyenne 4/π fois plus d'arêtes que sa longueur)
        edges = np.count_nonzero(np.diff(binary, axis=0)) + np.count_nonzero(np.diff(binary, axis=1))
        interfaces = 0.25 * np.pi * edges / pb.n
        text = f"périmètre {interfaces:.2f}"
    return {
        "rho": rho, "u_pinn": u_pinn, "u_exact": u_exact, "J_exact": j_exact, "J_binary": j_binary,
        "vol_binary": float(binary.mean()), "interfaces": interfaces, "interfaces_text": text,
        "grey_floor": 4.0 * eps * interfaces,   # gris d'un profil sigmoïde(d / ε) par unité d'interface
        "u_error": float(np.sqrt(np.mean((u_pinn - u_exact) ** 2) / np.mean(u_exact ** 2))),
    }


# --- données des figures (plotphase.py) ---
def save_npz(path: Path, data: dict):
    """Écriture atomique : un lecteur (plotphase.py --watch, copie distante) ne voit jamais un fichier partiel."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **{k: v.astype(np.float32) if isinstance(v, np.ndarray) and v.ndim == 2 and v.dtype == float
                       else v for k, v in data.items()})
    os.replace(tmp, path)


# --- programme ---
class _Formatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=_Formatter)
    g = p.add_argument_group("problème : -div(A(ρ) ∇u) = f, u = 0 au bord, A = a2 + (a1 - a2) ρ / (1 + q (1 - ρ))")
    g.add_argument("--dim", type=int, default=1, choices=(1, 2))
    g.add_argument("--a1", type=float, default=1.0, help="conductivité de la phase ρ = 1")
    g.add_argument("--a2", type=float, default=0.01, help="conductivité de la phase ρ = 0")
    g.add_argument("--v0", type=float, default=0.5, help="volume cible de la phase ρ = 1")
    g.add_argument("--source", type=float, default=1.0, help="f")
    g.add_argument("--band", type=float, nargs=2, default=[0.2, 0.3], help="objectif J = ∫_bande u²")
    g = p.add_argument_group("loss : J/J_ref + β résidu + λ Van der Waals + μ volume")
    g.add_argument("--lam", type=float, default=1e-2, help="λ, Van der Waals")
    g.add_argument("--beta", type=float, default=1e2, help="β, résidu de l'EDP (forme forte)")
    g.add_argument("--mu", type=float, default=1e2, help="μ, volume")
    g = p.add_argument_group("continuation : ε géométrique de eps0 à eps-min ; q = q0 jusqu'au palier q-start, "
                             "puis linéaire jusqu'à q-max (0 : loi linéaire, a1/a2 - 1 : harmonique)")
    g.add_argument("--eps0", type=float, default=0.1)
    g.add_argument("--eps-min", type=float, default=None, help="défaut : 4e-3 en 1D, 8e-3 en 2D")
    g.add_argument("--n-stages", type=int, default=None, help="paliers ; défaut : 10 en 1D, 8 en 2D")
    g.add_argument("--q0", type=float, default=0.0)
    g.add_argument("--q-max", type=float, default=0.0)
    g.add_argument("--q-start", type=int, default=0, help="palier (compté depuis 0) après lequel q monte")
    g = p.add_argument_group("discrétisation")
    g.add_argument("--elements", type=int, default=None, help="éléments par direction ; défaut 100")
    g.add_argument("--gauss", type=int, default=None, help="points de Gauss par élément ; défaut : 10 en 1D, 3 en 2D")
    g.add_argument("--width", type=int, default=None, help="neurones par couche ; défaut : 20 en 1D, 32 en 2D")
    g.add_argument("--depth", type=int, default=3, help="couches cachées tanh")
    g.add_argument("--n-exact", type=int, default=4000, help="1D : cellules de la solution exacte (diagnostics)")
    g.add_argument("--n-grid", type=int, default=200, help="2D : cellules par direction des volumes finis")
    g.add_argument("--float32", action="store_true", help="~5x plus rapide sur GPU grand public, moins précis")
    g = p.add_argument_group("optimisation (NOM:ITER enchaînés : adam, lbfgs, ng = gradient naturel)")
    g.add_argument("--opt0", nargs="*", default=["adam:2000"], metavar="NOM:ITER", help="premier palier, avant --opt")
    g.add_argument("--opt", nargs="+", default=["lbfgs:1000"], metavar="NOM:ITER", help="à chaque palier")
    g.add_argument("--lr", type=float, default=1e-3, help="pas d'Adam")
    g.add_argument("--lbfgs-memory", type=int, default=50, help="historique de L-BFGS")
    g.add_argument("--chunk", type=int, default=1024, help="points par paquet pour JᵀJ (ng)")
    g.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("références sans réseau (2D : optimisation par densité, filtre + RAMP + adjoint)")
    g.add_argument("--no-references", action="store_true", help="ne pas les calculer")
    g.add_argument("--ref-cells", type=int, default=100, help="2D : cellules par direction")
    g.add_argument("--ref-filter", type=float, default=0.03, help="2D : rayon du filtre de densité")
    g.add_argument("--ref-q-penal", type=float, default=300.0, help="2D : q final du design pénalisé")
    g.add_argument("--ref-q-steps", type=int, default=6, help="2D : paliers de q")
    g.add_argument("--ref-iters", type=int, default=300, help="2D : itérations L-BFGS-B par palier")
    g = p.add_argument_group("sorties")
    g.add_argument("--name", default="", help="nom du run (défaut : 1d ou 2d)")
    g.add_argument("--out", default="runs", help="dossier des runs")
    g.add_argument("--plot-every", type=int, default=None,
                   help="itérations entre deux mises à jour de live.npz ; défaut : 100 en 1D, 200 en 2D")
    a = p.parse_args(argv)
    for k, v in DIM_DEFAULTS.items():
        if getattr(a, k) is None:
            setattr(a, k, v[a.dim - 1])
    return a


def main(argv=None):
    a = parse(argv)
    if a.float32:
        jax.config.update("jax_enable_x64", False)
    pb = Problem(a)
    name = a.name or f"{a.dim}d"
    run_dir = Path(a.out) / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}_s{a.seed}"
    (run_dir / "params").mkdir(parents=True)
    latest = Path(a.out) / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)
    (run_dir / "config.json").write_text(json.dumps(vars(a), indent=2))
    log_file = open(run_dir / "log.txt", "w", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"run : {run_dir}   appareil : {jax.devices()[0]}")
    log(f"suivi : python plotphase.py {run_dir} --watch"
        + ("" if plotphase is None else f"   (ou l'image {run_dir / 'live.png'})"))
    x_el = np.linspace(0.0, 1.0, a.elements + 1)
    if not all(np.isclose(x_el, b).any() for b in pb.band):
        log(f"attention : les bornes de la bande {pb.band} ne tombent pas sur des bords d'éléments")
    log(f"J_ref (ρ = V0 partout, loi linéaire) = {pb.j_ref:.6e}   points de quadrature : {a.elements * a.gauss}"
        + ("²" if a.dim == 2 else ""))
    refs = None if a.no_references else references(pb, a, log)

    key = jax.random.PRNGKey(a.seed)
    ku, kr = jax.random.split(key)
    theta, unravel = ravel_pytree((init_mlp(ku, a.dim, a.width, a.depth),
                                   init_rho(kr, a.dim, a.width, a.depth, a.v0)))
    loss = Loss(pb, a, unravel)
    hist = {"stats": [], "eps": [], "stage_starts": []}
    checks, stages = [], []
    columns = ("iter", "J_exact", "J_binary", "vol_binary", "grey_floor")
    t0 = time.time()
    stage, eps, q, phase_name = 0, stage_eps(a, 0), stage_q(a, 0), "init"

    def report(final=False, status=""):
        snap = evaluate(pb, theta, unravel, eps, q)
        n = len(hist["stats"])
        checks.append({"iter": n, **{k: snap[k] for k in columns[1:]}})
        s = hist["stats"][-1] if n else np.zeros(6)
        if not final:
            log(f"[palier {stage + 1} {phase_name} | it {n:6d}] L {s[:4].sum():.4e}  J/Jref {s[0]:.4e}  "
                f"β·rés {s[1]:.2e}  λ·VdW {s[2]:.2e}  μ·vol {s[3]:.2e}  vol {s[4]:.4f}  gris {s[5]:.3f} | "
                f"J exact {snap['J_exact']:.4f}  J seuillé {snap['J_binary']:.4f}  erreur u {snap['u_error']:.1e}")
        summary = "\n".join([
            f"A = {a.a2:g} + {a.a1 - a.a2:g} ρ / (1 + q (1 - ρ))   V0 = {a.v0:g}   bande {list(pb.band)}",
            f"q (RAMP) : {a.q0:g} -> {a.q_max:g} (monte après le palier {a.q_start + 1})",
            f"λ = {a.lam:g}   β = {a.beta:g}   μ = {a.mu:g}",
            f"ε : {a.eps0:g} -> {a.eps_min:g} ({a.n_stages} paliers)",
            f"points : ({a.elements} x {a.gauss}){'²' if a.dim == 2 else ''} Gauss   "
            f"réseaux : {a.width} x {a.depth} tanh",
            f"optimiseurs : palier 1 {' '.join(a.opt0)}, puis {' '.join(a.opt)}",
            "",
            f"J PINN / J_ref       {s[0]:.4f}",
            f"J réf. / J_ref       {snap['J_exact']:.4f}   ({'exacte' if a.dim == 1 else 'volumes finis'})",
            f"J seuillé / J_ref    {snap['J_binary']:.4f}",
            *([f"référence binaire    {refs['binary']['J']:.4f}",
               f"référence relaxée    {refs['relaxed']['J']:.4f}"] if refs else []),
            "",
            f"∫ρ                   {s[4]:.4f}   (V0 = {a.v0:g})",
            f"volume seuillé       {snap['vol_binary']:.4f}",
            f"gris G               {s[5]:.4f}",
            f"plancher 4ε·n        {snap['grey_floor']:.4f}   ({snap['interfaces_text']})",
            f"erreur L2 sur u      {snap['u_error']:.2e}",
        ])
        title = (f"{name}   seed {a.seed}   palier {stage + 1}/{a.n_stages}   ε = {eps:.2e}   q = {q:g}   "
                 f"{phase_name}   it {n}   {time.time() - t0:.0f} s")
        data = {
            "dim": a.dim, "rho": snap["rho"], "u_pinn": snap["u_pinn"], "u_exact": snap["u_exact"],
            "stats": np.asarray(hist["stats"]).reshape(-1, 6), "eps_hist": np.asarray(hist["eps"]),
            "checks": np.asarray([[c[col] for col in columns] for c in checks], dtype=float),
            "stage_starts": np.asarray(hist["stage_starts"], dtype=int), "band": np.asarray(pb.band), "v0": a.v0,
            "title": title, "summary": summary, "done": final, "status": status,
        }
        data["x"] = pb.mid
        if refs:
            data["ref_binary_J"], data["ref_relaxed_J"] = refs["binary"]["J"], refs["relaxed"]["J"]
            data["ref_binary_rho"], data["ref_relaxed_rho"] = refs["binary"]["rho"], refs["relaxed"]["rho"]
            if a.dim == 1:
                data["ref_binary_edges"], data["ref_relaxed_x"] = refs["binary"]["edges"], refs["relaxed"]["x"]
        save_npz(run_dir / "live.npz", data)
        if plotphase is not None:
            fig = plotphase.draw(data, run_dir / ("final.png" if final else "live.png"))
            plotphase.plt_close(fig)
        return snap

    def run_phase(phase, hp):
        nonlocal theta, phase_name
        phase_name, _, count = phase.partition(":")
        count = int(count)
        if phase_name in ("adam", "lbfgs"):
            init, chunk = adam_runner(loss, a.lr) if phase_name == "adam" else lbfgs_runner(loss, a.lbfgs_memory)
            state = init(theta)
            done = 0
            while done < count:
                n = min(a.plot_every, count - done)
                theta, state, stats = chunk(theta, state, n, hp)
                hist["stats"] += list(np.asarray(stats))
                hist["eps"] += [eps] * n
                done += n
                report()
        elif phase_name == "ng":
            ng = NaturalGradient(loss)
            every = max(1, a.plot_every // 20)
            for k in range(count):
                theta, ok = ng.step(theta, hp)
                hist["stats"].append(np.asarray(loss.stats(theta, hp)[1]))
                hist["eps"].append(eps)
                if not ok:
                    log(f"ng : plus de descente (amortissement {ng.damping:.0e}), arrêt")
                    report()
                    break
                if (k + 1) % every == 0 or k + 1 == count:
                    report()
        else:
            raise ValueError(f"optimiseur inconnu : {phase_name!r} (adam, lbfgs, ng)")

    status = "terminé"
    try:
        for stage in range(a.n_stages):
            eps, q = stage_eps(a, stage), stage_q(a, stage)
            hp = jnp.array([eps, q])
            hist["stage_starts"].append(len(hist["stats"]))
            log(f"===== palier {stage + 1}/{a.n_stages}   ε = {eps:.3e}   q = {q:g} =====")
            t_stage = time.time()
            for phase in (a.opt0 if stage == 0 else []) + a.opt:
                run_phase(phase, hp)
            snap = evaluate(pb, theta, unravel, eps, q)
            s = hist["stats"][-1]
            stages.append({"stage": stage, "eps": eps, "q": q, "iters": len(hist["stats"]),
                           "time_s": time.time() - t_stage,
                           **dict(zip((*Loss.TERMS, "vol", "grey"), map(float, s), strict=True)),
                           **{k: snap[k] for k in ("J_exact", "J_binary", "vol_binary", "interfaces",
                                                   "grey_floor", "u_error")}})
            log(f"fin du palier {stage + 1} ({stages[-1]['time_s']:.1f} s) : J PINN {s[0]:.4f}  "
                f"J exact {snap['J_exact']:.4f}  J seuillé {snap['J_binary']:.4f} (vol {snap['vol_binary']:.3f})  "
                f"gris {s[5]:.3f} (plancher {snap['grey_floor']:.3f})  {snap['interfaces_text']}  "
                f"erreur u {snap['u_error']:.2e}")
            np.savez(run_dir / "params" / f"stage_{stage:02d}.npz", theta=np.asarray(theta), eps=eps, q=q)
    except KeyboardInterrupt:
        status = "interrompu"
        log("interrompu : sauvegarde de l'état courant")
    np.savez(run_dir / "params" / "final.npz", theta=np.asarray(theta), eps=eps, q=q)
    report(final=True, status=status)
    metrics = {
        "status": status, "time_s": time.time() - t0, "j_ref": pb.j_ref,
        **({"binary_reference": {k: refs["binary"][k] for k in ("J", "label")},
            "relaxed_reference": {k: refs["relaxed"][k] for k in ("J", "grey", "label")}} if refs else {}),
        "stages": stages, "final": stages[-1] if stages else None,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    log(f"{status} en {metrics['time_s']:.0f} s   -> {run_dir}")


if __name__ == "__main__":
    main()
