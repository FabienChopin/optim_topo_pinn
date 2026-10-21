"""Banc d'essai 2D : retrouver par PINN un design binaire à N inclusions disques (voir README.md).

Exemples :
    python pinn2d.py                                       # 1 disque, Adam puis L-BFGS
    python pinn2d.py --n-disks 4                           # 4 disques sur une grille décalée
    python pinn2d.py --disks 0.3 0.35 0.15 0.68 0.6 0.2 --opt adam:2000 lbfgs:2000 ng:50
    python pinn2d.py --n-disks 4 --capacity-only           # le réseau peut-il représenter la cible ?

Pendant le calcul, runs2d/latest/live.npz contient l'état courant (ρ, u, loss, topologie). Les figures en
sont tirées par plot2d.py, ici ou sur une autre machine (python plot2d.py --help). Si matplotlib est installé
là où le calcul tourne, live.png et capacity.png sont aussi écrits sur place.
À la fin : runs2d/<date>_<nom>/ contient live.npz, capacity.npz, log.txt, metrics.json, config.json, params.npz.
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
from scipy import ndimage
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

jax.config.update("jax_enable_x64", True)
try:   # figures sur place si matplotlib est installé ; sinon, python plot2d.py ailleurs
    import plot2d
    plot2d.use_files()
except ImportError:
    plot2d = None

PLOT_STEP = 2         # données des figures : une cellule fine sur PLOT_STEP dans chaque direction
CAP_RHO_TOL = 1e-2    # capacité : erreur L2 relative sur rho_eps
CAP_U_TOL = 1e-2      # capacité : erreur H1 relative sur u_eps
EVAL_BATCH = 8192     # points par paquet pour évaluer les réseaux sur la grille fine


# --- problème ---
class Problem:
    """-div(A(ρ) ∇u) = f sur ]0,1[², u = 0 au bord ; ρ* = 1 dans les disques, 0 ailleurs.
    Solution de référence : éléments finis P1 sur la grille fine (n x n carrés coupés en deux triangles)."""

    def __init__(self, a):
        self.a1, self.a2, self.q, self.f, self.eps = a.a1, a.a2, a.q, a.source, a.eps
        if a.disks:
            if len(a.disks) % 3:
                raise SystemExit("--disks attend des triplets cx cy r")
            d = np.array(a.disks, dtype=float).reshape(-1, 3)
        else:
            # grille k x k décalée : jamais de centre en (0.5, 0.5), où le MLP initial (biais nuls) est symétrique
            k = int(np.ceil(np.sqrt(a.n_disks)))
            s = 0.8 / k
            d = np.array([[0.1 + s * (i % k + 0.56), 0.1 + s * (i // k + 0.53), 0.3 * s] for i in range(a.n_disks)])
        self.centers, self.radii = d[:, :2], d[:, 2]
        self._check_disks()
        self.n = a.fine
        self.h = 1.0 / self.n
        self.nodes = np.linspace(0.0, 1.0, self.n + 1)
        self.mid = 0.5 * (self.nodes[1:] + self.nodes[:-1])
        self.cells = grid_points(self.mid)        # (n², 2), indice i * n + j pour (x_i, y_j)
        self.node_pts = grid_points(self.nodes)   # ((n+1)², 2)
        self._assemble()
        self.rho_star = self.binary(self.cells).reshape(self.n, self.n)
        self.target_betti = betti(self.rho_star > 0.5)
        self.volume = float(self.rho_star.mean())
        self.u_star = self.exact(self.rho_star)
        self.j_ref = self.l2sq(self.exact(np.full_like(self.rho_star, self.volume)) - self.u_star)
        self.u_eps = self.exact(self.profile(self.cells).reshape(self.n, self.n))
        t = [np.linspace(0.0, 2 * np.pi, max(16, int(np.ceil(2 * np.pi * r / (0.5 * self.h)))), endpoint=False)
             for r in self.radii]
        self.circle_pts = np.concatenate([c + r * np.stack([np.cos(s), np.sin(s)], 1)
                                          for c, r, s in zip(self.centers, self.radii, t, strict=True)])

    def _check_disks(self):
        c, r = self.centers, self.radii
        if np.any(r <= 0) or np.any(c - r[:, None] <= 0) or np.any(c + r[:, None] >= 1):
            raise SystemExit("chaque disque doit être strictement dans ]0,1[²")
        for i in range(len(r)):
            for j in range(i):
                if np.linalg.norm(c[i] - c[j]) <= r[i] + r[j]:
                    raise SystemExit(f"les disques {j} et {i} se touchent : la topologie cible serait ambiguë")

    def conductivity(self, rho):
        return self.a2 + (self.a1 - self.a2) * rho / (1.0 + self.q * (1.0 - rho))

    def sdist(self, x, y, xp=np):
        """Distance signée au bord de l'union des disques (> 0 dedans), exacte pour des disques disjoints.
        Le 1e-30 évite une dérivée NaN au centre exact d'un disque (point de Gauss au milieu d'un élément)."""
        d = [r - xp.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1e-30)
             for (cx, cy), r in zip(self.centers, self.radii, strict=True)]
        return d[0] if len(d) == 1 else xp.max(xp.stack(d), axis=0)

    def binary(self, pts):
        return (self.sdist(pts[:, 0], pts[:, 1]) >= 0).astype(float)

    def profile(self, pts, xp=np):
        """Profil de champ de phase cible : sigmoïde(distance signée / eps)."""
        return 1.0 / (1.0 + xp.exp(-self.sdist(pts[..., 0], pts[..., 1], xp) / self.eps))

    def _assemble(self):
        n, m = self.n, self.n + 1
        i, j = (g.ravel() for g in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
        v00, v10, v11, v01 = i * m + j, (i + 1) * m + j, (i + 1) * m + j + 1, i * m + j + 1
        tri = np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])   # 2 n² triangles
        k_low = 0.5 * np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=float)   # rigidités locales (A = 1)
        k_up = 0.5 * np.array([[1, 0, -1], [0, 1, -1], [-1, -1, 2]], dtype=float)
        self._kdata = np.concatenate([np.broadcast_to(k_low.ravel(), (n * n, 9)),
                                      np.broadcast_to(k_up.ravel(), (n * n, 9))])
        self._rows = np.repeat(tri, 3, axis=1).ravel()
        self._cols = np.tile(tri, (1, 3)).ravel()
        self._cell = np.concatenate([np.arange(n * n), np.arange(n * n)])
        area = 0.5 * self.h ** 2
        mass = area / 12.0 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float)
        shape = (m * m, m * m)
        self.mass = sp.coo_matrix((np.tile(mass.ravel(), 2 * n * n), (self._rows, self._cols)), shape).tocsr()
        self.lap = sp.coo_matrix((self._kdata.ravel(), (self._rows, self._cols)), shape).tocsr()
        self._load = self.f * np.bincount(tri.ravel(), minlength=m * m) * area / 3.0
        inner = np.zeros((m, m), dtype=bool)
        inner[1:-1, 1:-1] = True
        self._inner = np.flatnonzero(inner)

    def exact(self, rho_cells):
        """u aux nœuds ((n+1) x (n+1)) pour A constant par cellule de la grille fine (P1)."""
        m = self.n + 1
        a = self.conductivity(np.asarray(rho_cells).ravel())[self._cell]
        k = sp.coo_matrix(((a[:, None] * self._kdata).ravel(), (self._rows, self._cols)), (m * m, m * m)).tocsc()
        k = k[self._inner][:, self._inner]
        u = np.zeros(m * m)
        u[self._inner] = splu(k.tocsc(), permc_spec="MMD_AT_PLUS_A").solve(self._load[self._inner])
        return u.reshape(m, m)

    def l2sq(self, nodal):
        """int v^2 pour v P1 (valeurs aux nœuds)."""
        v = nodal.ravel()
        return float(v @ (self.mass @ v))

    def h1sq(self, nodal):
        v = nodal.ravel()
        return self.l2sq(nodal) + float(v @ (self.lap @ v))

    def interp(self, nodal, pts):
        """Interpolation P1 (triangles de la grille fine) aux points pts."""
        s = pts / self.h
        i = np.clip(np.floor(s[:, 0]).astype(int), 0, self.n - 1)
        j = np.clip(np.floor(s[:, 1]).astype(int), 0, self.n - 1)
        xi, eta = s[:, 0] - i, s[:, 1] - j
        u00, u10, u11, u01 = nodal[i, j], nodal[i + 1, j], nodal[i + 1, j + 1], nodal[i, j + 1]
        return np.where(xi >= eta, u00 + (u10 - u00) * xi + (u11 - u10) * eta,
                        u00 + (u11 - u01) * xi + (u01 - u00) * eta)

    def hausdorff(self, rho):
        """Distance de Hausdorff entre l'isoligne rho = 1/2 (grille fine) et les cercles cibles."""
        pts = contour_points(self.mid, rho)
        if not len(pts):
            return np.nan
        to_target = np.max(np.abs(self.sdist(pts[:, 0], pts[:, 1])))
        return float(max(to_target, cKDTree(pts).query(self.circle_pts)[0].max()))


def grid_points(t):
    x, y = np.meshgrid(t, t, indexing="ij")
    return np.stack([x.ravel(), y.ravel()], 1)


def gauss_composite(n_el, n_g):
    t, w = np.polynomial.legendre.leggauss(n_g)
    edges = np.linspace(0.0, 1.0, n_el + 1)
    half = 0.5 * np.diff(edges)[:, None]
    return (0.5 * (edges[:-1, None] + edges[1:, None]) + half * t).ravel(), (half * w).ravel()


def gauss_2d(n_el, n_g):
    """Gauss composite tensorisé : (n_el n_g)² points."""
    x, w = gauss_composite(n_el, n_g)
    return grid_points(x), np.outer(w, w).ravel()


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
def init_mlp(key, width, depth):
    sizes = [2] + [width] * depth + [1]
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
    return x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]) * mlp(pu, x)


def rho_fn(pr, x):
    return jax.nn.sigmoid(mlp(pr, x))


def init_rho(key, width, depth, value):
    p = init_mlp(key, width, depth)
    w, _ = p[-1]
    p[-1] = (0.1 * w, jnp.full(1, np.log(value / (1.0 - value))))
    return p


@jax.jit
def _on_grid(p, pts):
    return jax.lax.map(lambda x: mlp(p, x), pts, batch_size=EVAL_BATCH)


def rho_grid(pr, pts):
    return np.asarray(jax.nn.sigmoid(_on_grid(pr, jnp.asarray(pts))))


def u_grid(pu, pts):
    return pts[:, 0] * (1.0 - pts[:, 0]) * pts[:, 1] * (1.0 - pts[:, 1]) * np.asarray(_on_grid(pu, jnp.asarray(pts)))


def divergence(conductivity, pu, x):
    """div(A ∇u) en x, A = conductivity(x)."""
    flux = lambda y: conductivity(y) * jax.grad(u_fn, 1)(pu, y)  # noqa: E731
    return jnp.trace(jax.jacfwd(flux)(x))


# --- loss de moindres carrés ---
class Loss:
    TERMS = ("J", "residual", "vdw", "volume")

    def __init__(self, pb: Problem, a, unravel):
        x, w = gauss_2d(a.elements, a.gauss)
        data = {"x": x, "sw": np.sqrt(w), "u_star": pb.interp(pb.u_star, x)}
        self.data = {k: jnp.asarray(v) for k, v in data.items()}
        self.batches = batches(a.chunk, **data)
        self.w = jnp.asarray(w)
        self.pb, self.a, self.unravel = pb, a, unravel

    def local(self, theta, d):
        """Résidus ponctuels (points, 5) : J, résidu de l'EDP, Van der Waals (∂x ρ, ∂y ρ, double puits)."""
        pb, a = self.pb, self.a
        pu, pr = self.unravel(theta)

        def point(x):
            rho, drho = jax.value_and_grad(rho_fn, 1)(pr, x)
            div = divergence(lambda y: pb.conductivity(rho_fn(pr, y)), pu, x)
            return u_fn(pu, x), rho, drho, div

        u, rho, drho, div = jax.vmap(point)(d["x"])
        sw = d["sw"]
        return jnp.stack([
            sw * (u - d["u_star"]) / np.sqrt(pb.j_ref),
            np.sqrt(a.beta) * sw * (-div - pb.f),
            np.sqrt(a.lam * pb.eps) * sw * drho[:, 0],
            np.sqrt(a.lam * pb.eps) * sw * drho[:, 1],
            np.sqrt(a.lam / pb.eps) * sw * rho * (1.0 - rho),
        ], 1)

    def glob(self, theta):
        """Résidu global : volume."""
        if self.a.mu == 0:
            return jnp.zeros(1)
        rho = jax.vmap(rho_fn, (None, 0))(self.unravel(theta)[1], self.data["x"])
        return np.sqrt(self.a.mu) * jnp.atleast_1d(jnp.sum(self.w * rho) - self.pb.volume)

    def value_terms(self, theta):
        """L et ses termes (J, résidu, Van der Waals, volume), d'un seul calcul des résidus."""
        r2 = self.local(theta, self.data) ** 2
        terms = jnp.stack([r2[:, 0].sum(), r2[:, 1].sum(), r2[:, 2:].sum(), jnp.sum(self.glob(theta) ** 2)])
        return terms.sum(), terms

    def terms(self, theta):
        return self.value_terms(theta)[1]

    def value(self, theta):
        return self.value_terms(theta)[0]


# --- optimiseurs ---
def adam_runner(loss: Loss, lr):
    opt = optax.adam(lr)

    @partial(jax.jit, static_argnums=2)
    def chunk(theta, state, n):
        def body(carry, _):
            th, st = carry
            (_, terms), g = jax.value_and_grad(loss.value_terms, has_aux=True)(th)
            up, st = opt.update(g, st)
            return (th + up, st), terms
        (theta, state), terms = jax.lax.scan(body, (theta, state), None, length=n)
        return theta, state, terms
    return opt.init, chunk


def lbfgs_runner(loss: Loss, memory):
    opt = optax.lbfgs(memory_size=memory)
    vg = optax.value_and_grad_from_state(loss.value)

    @partial(jax.jit, static_argnums=2)
    def chunk(theta, state, n):
        def body(carry, _):
            th, st = carry
            v, g = vg(th, state=st)
            up, st = opt.update(g, st, th, value=v, grad=g, value_fn=loss.value)
            th = optax.apply_updates(th, up)
            return (th, st), loss.terms(th)
        (theta, state), terms = jax.lax.scan(body, (theta, state), None, length=n)
        return theta, state, terms
    return opt.init, chunk


class LevenbergMarquardt:
    """Gauss-Newton amorti sur L = |r(θ)|^2 : (JᵀJ + μ max diag(JᵀJ) I) d = Jᵀr, θ ← θ - d.
    Gain = baisse réelle / baisse prédite |r|^2 - |r - J d|^2 : refus et μ × 10 si < 1/4, μ / 10 si > 3/4.
    r = résidus ponctuels local(θ, paquet) sur chaque paquet de points, plus des résidus globaux glob(θ) :
    JᵀJ (taille θ²) est accumulé paquet par paquet, la Jacobienne complète n'est jamais formée."""

    def __init__(self, local, batches, glob=None, damping=1e-8, min_damping=1e-12, max_tries=15):
        self.damping, self.min_damping, self.max_tries = damping, min_damping, max_tries
        flat = lambda th, b: local(th, b).ravel()  # noqa: E731

        def total(theta):
            def body(acc, b):
                r = flat(theta, b)
                return acc + r @ r, None
            sq, _ = jax.lax.scan(body, 0.0, batches)
            return sq if glob is None else sq + jnp.sum(glob(theta) ** 2)

        @jax.jit
        def prepare(theta):
            p = theta.size

            def body(carry, b):
                gram, g, sq = carry
                r = flat(theta, b)
                jac = (jax.jacrev if r.size < p else jax.jacfwd)(flat)(theta, b)
                return (gram + jac.T @ jac, g + jac.T @ r, sq + r @ r), None
            (gram, g, sq), _ = jax.lax.scan(body, (jnp.zeros((p, p)), jnp.zeros(p), 0.0), batches)
            if glob is not None:
                rg, jg = glob(theta), jax.jacrev(glob)(theta)
                gram, g, sq = gram + jg.T @ jg, g + jg.T @ rg, sq + rg @ rg
            return sq, gram, g

        @jax.jit
        def trial(theta, gram, g, mu):
            scale = mu * jnp.max(jnp.diag(gram))
            chol = jnp.linalg.cholesky(gram + scale * jnp.eye(len(g)))
            d = jax.scipy.linalg.cho_solve((chol, True), g)
            predicted = 2.0 * g @ d - d @ gram @ d
            new = theta - d
            return new, total(new), predicted

        self.prepare, self.trial = prepare, trial

    def step(self, theta):
        """Un pas ; rend (θ, accepté)."""
        l0, gram, g = self.prepare(theta)
        for _ in range(self.max_tries):
            new, l1, predicted = self.trial(theta, gram, g, self.damping)
            gain = float((l0 - l1) / predicted)
            if np.isfinite(gain) and gain > 0.25:
                if gain > 0.75:
                    self.damping = max(self.damping / 10.0, self.min_damping)
                return new, True
            self.damping *= 10.0
        return theta, False


# --- diagnostics ---
def betti(mask):
    """(composantes de mask en 8-connexité, trous = composantes du complémentaire, 4-connexité, sans le bord)."""
    _, b0 = ndimage.label(mask, structure=np.ones((3, 3)))
    lab, nb = ndimage.label(~mask)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return int(b0), int(nb - np.count_nonzero(border))


def contour_points(t, rho):
    """Points de l'isoligne rho = 1/2, interpolés le long des arêtes de la grille t x t (rho[i, j] en (t_i, t_j))."""
    s = rho - 0.5
    out = []
    for axis in (0, 1):
        a, b = (s[:-1], s[1:]) if axis == 0 else (s[:, :-1], s[:, 1:])
        i, j = np.nonzero(np.sign(a) != np.sign(b))
        frac = a[i, j] / (a[i, j] - b[i, j])
        x, y = (t[i] + frac * (t[i + 1] - t[i]), t[j]) if axis == 0 else (t[i], t[j] + frac * (t[j + 1] - t[j]))
        out.append(np.stack([x, y], 1))
    return np.concatenate(out)


def evaluate(pb: Problem, theta, unravel):
    pu, pr = unravel(theta)
    rho = rho_grid(pr, pb.cells).reshape(pb.n, pb.n)
    u = u_grid(pu, pb.node_pts).reshape(pb.n + 1, pb.n + 1)
    u_ex = pb.exact(rho)
    b0, b1 = betti(rho > 0.5)
    return {
        "rho": rho, "u": u, "u_exact": u_ex,
        "J_exact": pb.l2sq(u_ex - pb.u_star) / pb.j_ref,
        "u_error": float(np.sqrt(pb.l2sq(u - u_ex) / pb.l2sq(u_ex))),
        "rho_error": float(np.sqrt(np.mean((rho - pb.rho_star) ** 2))),
        "components": b0, "holes": b1, "hausdorff": pb.hausdorff(rho),
    }


# --- test de capacité ---
def capacity(pb: Problem, a, key, log):
    """rho : moindres carrés sur rho_eps ; u : forme forte à rho = rho_eps fixé. Levenberg-Marquardt."""
    x, w = gauss_2d(a.elements, a.gauss)
    bt = batches(a.chunk, x=x, sw=np.sqrt(w), target=pb.profile(x))
    k1, k2 = jax.random.split(key)
    theta_r, unr = ravel_pytree(init_rho(k1, a.width, a.depth, pb.volume))
    theta_u, unu = ravel_pytree(init_mlp(k2, a.width, a.depth))

    def res_rho(t, b):
        return b["sw"] * (jax.vmap(rho_fn, (None, 0))(unr(t), b["x"]) - b["target"])

    def res_u(t, b):
        a_eps = lambda y: pb.conductivity(pb.profile(y, jnp))  # noqa: E731
        div = jax.vmap(partial(divergence, a_eps, unu(t)))(b["x"])
        return b["sw"] * (-div - pb.f)

    rho_eps = pb.profile(pb.cells).reshape(pb.n, pb.n)

    def rho_error(t):
        rho = rho_grid(unr(t), pb.cells).reshape(pb.n, pb.n)
        return float(np.sqrt(np.mean((rho - rho_eps) ** 2) / np.mean(rho_eps ** 2))), rho

    def u_error(t):
        u = u_grid(unu(t), pb.node_pts).reshape(pb.n + 1, pb.n + 1)
        return float(np.sqrt(pb.h1sq(u - pb.u_eps) / pb.h1sq(pb.u_eps))), u

    out = {}
    for name, res, theta, error, tol in (("rho", res_rho, theta_r, rho_error, CAP_RHO_TOL),
                                         ("u", res_u, theta_u, u_error, CAP_U_TOL)):
        lm = LevenbergMarquardt(res, bt)
        t0 = time.time()
        for k in range(a.capacity_iters):   # arrêt dès que l'erreur passe sous tol / 10 (contrôle tous les 20 pas)
            theta, ok = lm.step(theta)
            if not ok or ((k + 1) % 20 == 0 and error(theta)[0] < 0.1 * tol):
                break
        out[name] = (theta, time.time() - t0, k + 1)
    (rho_err, rho_fit), (u_err, u_fit) = rho_error(out["rho"][0]), u_error(out["u"][0])
    b0, b1 = betti(rho_fit > 0.5)
    res = {
        "rho_error": rho_err, "rho_components": b0, "rho_holes": b1, "rho_hausdorff": pb.hausdorff(rho_fit),
        "u_error_h1": u_err, "steps": [out["rho"][2], out["u"][2]], "time_s": out["rho"][1] + out["u"][1],
    }
    res["ok"] = bool(res["rho_error"] < CAP_RHO_TOL and (b0, b1) == pb.target_betti and res["u_error_h1"] < CAP_U_TOL)
    log(f"capacité ({res['time_s']:.1f} s, {res['steps'][0]} + {res['steps'][1]} pas) : "
        f"rho erreur L2 rel {res['rho_error']:.2e}, {b0}/{pb.target_betti[0]} composantes, "
        f"{b1}/{pb.target_betti[1]} trous, Hausdorff {res['rho_hausdorff']:.1e} ; "
        f"u (forme forte, rho_eps fixé) erreur H1 rel {res['u_error_h1']:.2e} "
        f"-> {'OK' if res['ok'] else 'HORS DE PORTÉE'}")
    k = PLOT_STEP
    data = {"rho_eps": rho_eps[::k, ::k], "rho_fit": rho_fit[::k, ::k], "u_eps": pb.u_eps[::k, ::k],
            "u_fit": u_fit[::k, ::k], "centers": pb.centers, "radii": pb.radii,
            "rho_error": res["rho_error"], "u_error_h1": res["u_error_h1"],
            "title": f"test de capacité : {a.width} x {a.depth} tanh, ε = {pb.eps:g}, {len(pb.radii)} "
                     f"disque(s) -> {'OK' if res['ok'] else 'HORS DE PORTÉE'}"}
    return res, data


# --- données des figures (plot2d.py) ---
def save_npz(path: Path, data: dict):
    """Écriture atomique : un lecteur (plot2d.py --watch, copie distante) ne voit jamais un fichier partiel."""
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
    g = p.add_argument_group("cible (ρ* = 1 dans les disques, 0 ailleurs ; u* exact)")
    g.add_argument("--n-disks", type=int, default=1, metavar="N",
                   help="nombre de disques, sur une grille décalée (jamais centrée en (0.5, 0.5))")
    g.add_argument("--disks", type=float, nargs="+", default=None, metavar="CX CY R",
                   help="disques choisis, en triplets centre x, centre y, rayon (remplace --n-disks)")
    g = p.add_argument_group("physique : -div(A(ρ) ∇u) = f, u = 0 au bord, A = a2 + (a1 - a2) ρ / (1 + q (1 - ρ))")
    g.add_argument("--a1", type=float, default=1.0, help="conductivité de la phase ρ = 1 (disques)")
    g.add_argument("--a2", type=float, default=0.01, help="conductivité de la phase ρ = 0 (matrice)")
    g.add_argument("--q", type=float, default=99.0, help="RAMP ; a1/a2 - 1 = loi harmonique (gris non favorisé)")
    g.add_argument("--source", type=float, default=1.0, help="f")
    g = p.add_argument_group("loss : J/J_ref + β résidu + λ Van der Waals + μ volume")
    g.add_argument("--eps", type=float, default=1e-2, help="ε, largeur d'interface (≈ 4.4 ε de 10 à 90 %%)")
    g.add_argument("--lam", type=float, default=1e-3, help="λ, Van der Waals")
    g.add_argument("--beta", type=float, default=1e2, help="β, résidu de l'EDP (forme forte)")
    g.add_argument("--mu", type=float, default=0.0, help="μ, volume")
    g = p.add_argument_group("discrétisation")
    g.add_argument("--elements", type=int, default=50, help="éléments par direction (Gauss composite tensorisé)")
    g.add_argument("--gauss", type=int, default=6, help="points de Gauss par élément et par direction")
    g.add_argument("--width", type=int, default=40, help="neurones par couche (réseaux u et ρ)")
    g.add_argument("--depth", type=int, default=4, help="couches cachées tanh")
    g.add_argument("--fine", type=int, default=400, help="cellules par direction de la solution de référence (EF P1)")
    g = p.add_argument_group("optimisation")
    g.add_argument("--opt", nargs="+", default=["adam:2000", "lbfgs:2000"], metavar="NOM:ITER",
                   help="optimiseurs enchaînés : adam, lbfgs, ng (Gauss-Newton amorti)")
    g.add_argument("--lr", type=float, default=1e-3, help="pas d'Adam")
    g.add_argument("--lbfgs-memory", type=int, default=50, help="historique de L-BFGS")
    g.add_argument("--chunk", type=int, default=1024, help="points par paquet pour JᵀJ (ng, test de capacité)")
    g.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("test de capacité (avant l'entraînement)")
    g.add_argument("--capacity-only", action="store_true", help="faire le test puis s'arrêter")
    g.add_argument("--no-capacity", action="store_true", help="sauter le test")
    g.add_argument("--capacity-iters", type=int, default=300, help="pas de Gauss-Newton maximum par réseau")
    g = p.add_argument_group("sorties")
    g.add_argument("--name", default="", help="nom du run (défaut : d<nombre de disques>)")
    g.add_argument("--out", default="runs2d", help="dossier des runs")
    g.add_argument("--plot-every", type=int, default=200,
                   help="itérations entre deux mises à jour de live.npz (ng : tous les plot_every/20 pas)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    pb = Problem(a)
    name = a.name or f"d{len(pb.radii)}"
    run_dir = Path(a.out) / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}_s{a.seed}"
    run_dir.mkdir(parents=True)
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

    disks = [[round(float(v), 4) for v in (*c, r)] for c, r in zip(pb.centers, pb.radii, strict=True)]
    log(f"run : {run_dir}   appareil : {jax.devices()[0]}")
    log(f"suivi : python plot2d.py {run_dir} --watch"
        + ("" if plot2d is None else f"   (ou l'image {run_dir / 'live.png'})"))
    log(f"cible : disques (cx, cy, r) {disks}, (composantes, trous) = {pb.target_betti}, "
        f"V* = {pb.volume:.4f}, J_ref = {pb.j_ref:.4e}")
    if pb.h > 0.4 * a.eps:   # Hausdorff mesuré à ~0.65 h près : au-delà, plus d'un quart de la tolérance ε
        log(f"attention : grille de référence trop grossière pour ε = {a.eps:g} (h = {pb.h:.1e}, Hausdorff mesuré "
            f"à ~{0.65 * pb.h:.1e} près) ; prendre --fine {int(np.ceil(2.5 / a.eps))} ou plus")
    key = jax.random.PRNGKey(a.seed)
    metrics = {}
    if not a.no_capacity:
        log("test de capacité : ajustement des réseaux sur la cible")
        cap, data = capacity(pb, a, jax.random.fold_in(key, 1), log)
        save_npz(run_dir / "capacity.npz", data)
        if plot2d is not None:
            plot2d.draw_capacity(data, run_dir / "capacity.png")
        metrics["capacity"] = cap
        if a.capacity_only:
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            return

    ku, kr = jax.random.split(key)
    theta, unravel = ravel_pytree((init_mlp(ku, a.width, a.depth), init_rho(kr, a.width, a.depth, pb.volume)))
    loss = Loss(pb, a, unravel)
    hist = {"terms": [], "phase_starts": []}
    checks = []
    columns = ("iter", "J_exact", "u_error", "rho_error", "components", "holes", "hausdorff")
    t0 = time.time()
    phase_name = "init"

    def report(done=False, verdict=""):
        snap = evaluate(pb, theta, unravel)
        n = len(hist["terms"])
        checks.append({"iter": n, **{k: snap[k] for k in columns[1:]}})
        terms = hist["terms"][-1] if n else np.zeros(4)
        if not done:
            log(f"[{phase_name} | it {n:6d}] L {terms.sum():.3e}  J {terms[0]:.2e}  rés {terms[1]:.2e}  "
                f"VdW {terms[2]:.2e}  vol {terms[3]:.1e} | J exact {snap['J_exact']:.2e}  "
                f"erreur u {snap['u_error']:.1e}  ρ-ρ* {snap['rho_error']:.2e}  "
                f"{snap['components']}/{pb.target_betti[0]} composantes  {snap['holes']}/{pb.target_betti[1]} trous  "
                f"Hausdorff {snap['hausdorff']:.1e}")
        summary = "\n".join([
            f"cible : {len(pb.radii)} disque(s) (cx, cy, r)",
            *[f"  {d}" for d in disks[:6]], *(["  ..."] if len(disks) > 6 else []),
            f"A = {a.a2:g} + {a.a1 - a.a2:g} ρ / (1 + {a.q:g} (1 - ρ)),  f = {a.source:g}",
            f"ε = {a.eps:g}   λ = {a.lam:g}   β = {a.beta:g}   μ = {a.mu:g}",
            f"points : ({a.elements} x {a.gauss})² Gauss   réseaux : {a.width} x {a.depth} tanh",
            f"optimiseurs : {' '.join(a.opt)}",
            *([f"capacité : {'OK' if metrics['capacity']['ok'] else 'HORS DE PORTÉE'} "
               f"(ρ {metrics['capacity']['rho_error']:.1e}, u H1 {metrics['capacity']['u_error_h1']:.1e})"]
              if "capacity" in metrics else []),
            "",
            f"J PINN / J_ref        {terms[0]:.3e}",
            f"J exact / J_ref       {snap['J_exact']:.3e}",
            f"écart RMS ρ - ρ*      {snap['rho_error']:.3e}",
            f"composantes           {snap['components']} / {pb.target_betti[0]}",
            f"trous                 {snap['holes']} / {pb.target_betti[1]}",
            f"Hausdorff interfaces  {snap['hausdorff']:.2e}   (ε = {a.eps:g})",
            f"erreur relative u     {snap['u_error']:.2e}",
        ])
        title = f"{name}   seed {a.seed}   {phase_name}   it {n}   {time.time() - t0:.0f} s"
        k = PLOT_STEP
        data = {
            "rho": snap["rho"][::k, ::k], "u_star": pb.u_star[::k, ::k], "u_exact": snap["u_exact"][::k, ::k],
            "u": snap["u"][::k, ::k], "centers": pb.centers, "radii": pb.radii,
            "u_error": snap["u_error"], "terms": np.asarray(hist["terms"]).reshape(-1, 4),
            "checks": np.asarray([[c[col] for col in columns] for c in checks], dtype=float),
            "phase_starts": np.asarray(hist["phase_starts"], dtype=int), "target_betti": np.asarray(pb.target_betti),
            "eps": a.eps, "title": title, "summary": summary, "done": done, "verdict": verdict,
        }
        save_npz(run_dir / "live.npz", data)
        if plot2d is not None:
            fig = plot2d.draw_live(data, run_dir / ("final.png" if done else "live.png"))
            plot2d.plt_close(fig)
        return snap

    report()
    for phase in a.opt:
        phase_name, _, count = phase.partition(":")
        count = int(count)
        hist["phase_starts"].append(len(hist["terms"]))
        log(f"===== {phase_name} : {count} itérations =====")
        if phase_name in ("adam", "lbfgs"):
            init, chunk = adam_runner(loss, a.lr) if phase_name == "adam" else lbfgs_runner(loss, a.lbfgs_memory)
            state = init(theta)
            done = 0
            while done < count:
                n = min(a.plot_every, count - done)
                theta, state, terms = chunk(theta, state, n)
                hist["terms"] += list(np.asarray(terms))
                done += n
                report()
        elif phase_name == "ng":
            lm = LevenbergMarquardt(loss.local, loss.batches, loss.glob)
            every = max(1, a.plot_every // 20)
            for k in range(count):
                theta, ok = lm.step(theta)
                hist["terms"].append(np.asarray(loss.terms(theta)))
                if not ok:
                    log(f"ng : plus de descente (amortissement {lm.damping:.0e}), arrêt")
                    report()
                    break
                if (k + 1) % every == 0 or k + 1 == count:
                    report()
        else:
            raise ValueError(f"optimiseur inconnu : {phase_name!r} (adam, lbfgs, ng)")
    snap = evaluate(pb, theta, unravel)
    np.savez(run_dir / "params.npz", theta=np.asarray(theta))
    found = (snap["components"], snap["holes"]) == pb.target_betti and snap["hausdorff"] < a.eps
    verdict = (f"{'TOPOLOGIE RETROUVÉE' if found else 'TOPOLOGIE NON RETROUVÉE'} ({snap['components']}/"
               f"{pb.target_betti[0]} composantes, {snap['holes']}/{pb.target_betti[1]} trous, Hausdorff "
               f"{snap['hausdorff']:.2e} pour ε = {a.eps:g}, J exact {snap['J_exact']:.3e})")
    checks.pop()   # le dernier état est re-rapporté, marqué comme final
    report(done=True, verdict=verdict)
    metrics["final"] = {**checks[-1], "found": bool(found)}
    metrics["time_s"] = time.time() - t0
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    log(f"terminé en {metrics['time_s']:.0f} s : {verdict}")


if __name__ == "__main__":
    main()
