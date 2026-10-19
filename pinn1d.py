"""Banc d'essai 1D : retrouver par PINN un design binaire à N interfaces (voir README.md).

Exemples :
    python pinn1d.py                                       # 1 interface, Adam puis L-BFGS
    python pinn1d.py --n-interfaces 3                      # 3 interfaces réparties
    python pinn1d.py --interfaces 0.3 0.7 --opt adam:2000 lbfgs:2000 ng:200
    python pinn1d.py --n-interfaces 4 --capacity-only      # le réseau peut-il représenter la cible ?

Pendant le calcul, runs1d/latest/live.npz contient l'état courant (ρ, u, loss, topologie). Les figures en
sont tirées par plot1d.py, ici ou sur une autre machine (python plot1d.py --help). Si matplotlib est installé
là où le calcul tourne, live.png et capacity.png sont aussi écrits sur place.
À la fin : runs1d/<date>_<nom>/ contient live.npz, capacity.npz, log.txt, metrics.json, config.json, params.npz.
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
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)
try:   # figures sur place si matplotlib est installé ; sinon, python plot1d.py ailleurs
    import plot1d
    plot1d.use_files()
except ImportError:
    plot1d = None

FINE = 50000          # cellules de la solution exacte (diagnostics, cibles)
PLOT_STEP = 10        # données des figures : une cellule fine sur PLOT_STEP (5000 points)
CAP_RHO_TOL = 1e-2    # capacité : erreur L2 relative sur rho_eps
CAP_U_TOL = 1e-2      # capacité : erreur H1 relative sur u_eps


# --- problème ---
class Problem:
    def __init__(self, a):
        self.a1, self.a2, self.q, self.f, self.eps = a.a1, a.a2, a.q, a.source, a.eps
        if a.interfaces:
            xs = sorted(a.interfaces)
        else:
            # pas en 0.5 : à l'initialisation (biais nuls), le MLP tanh est impair en 2x - 1, rho croise 0.5 en 0.5
            xs = [0.1 + 0.8 * (k + 0.6) / a.n_interfaces for k in range(a.n_interfaces)]
        self.interfaces = np.array(xs, dtype=float)
        self.first = a.first_phase
        self.signs = np.array([(1 if self.first == 0 else -1) * (-1) ** k for k in range(len(xs))], dtype=float)
        self.edges = np.linspace(0.0, 1.0, FINE + 1)
        self.mid = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.rho_star = self.binary(self.mid)
        self.volume = float(self.rho_star.mean())
        self.u_star_nodes, self.du_star = self.exact(self.rho_star)
        u_hom, _ = self.exact(np.full(FINE, self.volume))
        self.j_ref = self.l2sq(u_hom - self.u_star_nodes)
        self.u_eps_nodes, self.du_eps = self.exact(self.profile(self.mid))

    def conductivity(self, rho):
        return self.a2 + (self.a1 - self.a2) * rho / (1.0 + self.q * (1.0 - rho))

    def binary(self, x):
        rho = np.full_like(x, float(self.first))
        for xk, s in zip(self.interfaces, self.signs, strict=True):
            rho = rho + s * (x >= xk)
        return rho

    def profile(self, x, xp=np):
        """Profil de champ de phase cible : sigmoïde((x - x_k) / eps) par interface."""
        rho = float(self.first) + 0.0 * x
        for xk, s in zip(self.interfaces, self.signs, strict=True):
            rho = rho + s / (1.0 + xp.exp(-(x - xk) / self.eps))
        return rho

    def exact(self, rho_cells):
        """u aux nœuds et u' aux milieux pour A constant par cellule (grille fine) : A u' = C - f x."""
        a = self.conductivity(rho_cells)
        h = np.diff(self.edges)
        sq = 0.5 * self.f * (self.edges[1:] ** 2 - self.edges[:-1] ** 2)
        c = np.sum(sq / a) / np.sum(h / a)
        u = np.concatenate([[0.0], np.cumsum((c * h - sq) / a)])
        return u, (c - self.f * self.mid) / a

    def l2sq(self, nodal):
        """int v^2 pour v linéaire par cellule (valeurs aux nœuds)."""
        h = np.diff(self.edges)
        return float(np.sum(h * (nodal[:-1] ** 2 + nodal[:-1] * nodal[1:] + nodal[1:] ** 2) / 3.0))

    def interp(self, nodal, x):
        return np.interp(x, self.edges, nodal)


def gauss_composite(n_el, n_g):
    t, w = np.polynomial.legendre.leggauss(n_g)
    edges = np.linspace(0.0, 1.0, n_el + 1)
    half = 0.5 * np.diff(edges)[:, None]
    return (0.5 * (edges[:-1, None] + edges[1:, None]) + half * t).ravel(), (half * w).ravel()


# --- réseaux ---
def init_mlp(key, width, depth):
    sizes = [1] + [width] * depth + [1]
    params = []
    for k, (m, n) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:], strict=True), strict=True):
        params.append((jax.random.normal(k, (n, m)) * np.sqrt(2.0 / (m + n)), jnp.zeros(n)))
    return params


def mlp(p, x):
    h = jnp.array([2.0 * x - 1.0])
    for w, b in p[:-1]:
        h = jnp.tanh(w @ h + b)
    w, b = p[-1]
    return (w @ h + b)[0]


def u_fn(pu, x):
    return x * (1.0 - x) * mlp(pu, x)


def rho_fn(pr, x):
    return jax.nn.sigmoid(mlp(pr, x))


def init_rho(key, width, depth, value):
    p = init_mlp(key, width, depth)
    w, _ = p[-1]
    p[-1] = (0.1 * w, jnp.full(1, np.log(value / (1.0 - value))))
    return p


# --- loss de moindres carrés ---
class Loss:
    TERMS = ("J", "residual", "vdw", "volume")

    def __init__(self, pb: Problem, a, unravel):
        x, w = gauss_composite(a.elements, a.gauss)
        self.x, self.sw = jnp.asarray(x), jnp.sqrt(jnp.asarray(w))
        self.w = jnp.asarray(w)
        self.u_star = jnp.asarray(pb.interp(pb.u_star_nodes, x))
        self.pb, self.a, self.unravel = pb, a, unravel

    def residuals(self, theta):
        """Vecteur r, L = |r|^2, et indices de fin de chaque terme."""
        pb, a = self.pb, self.a
        pu, pr = self.unravel(theta)

        def point(x):
            du = jax.grad(u_fn, 1)
            flux = lambda y: pb.conductivity(rho_fn(pr, y)) * du(pu, y)  # noqa: E731
            return u_fn(pu, x), rho_fn(pr, x), jax.grad(rho_fn, 1)(pr, x), jax.grad(flux)(x)

        u, rho, drho, dflux = jax.vmap(point)(self.x)
        return jnp.concatenate([
            self.sw * (u - self.u_star) / np.sqrt(pb.j_ref),
            np.sqrt(a.beta) * self.sw * (-dflux - pb.f),
            np.sqrt(a.lam * pb.eps) * self.sw * drho,
            np.sqrt(a.lam / pb.eps) * self.sw * rho * (1.0 - rho),
            jnp.sqrt(a.mu) * jnp.atleast_1d(jnp.sum(self.w * rho) - pb.volume),
        ])

    def value_terms(self, theta):
        """L et ses termes (J, résidu, Van der Waals, volume), d'un seul calcul des résidus."""
        r2 = self.residuals(theta) ** 2
        n = len(self.x)
        return r2.sum(), jnp.stack([r2[:n].sum(), r2[n:2 * n].sum(), r2[2 * n:4 * n].sum(), r2[4 * n:].sum()])

    def terms(self, theta):
        return self.value_terms(theta)[1]

    def value(self, theta):
        return jnp.sum(self.residuals(theta) ** 2)


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
    Gain = baisse réelle / baisse prédite |r|^2 - |r - J d|^2 : refus et μ × 10 si < 1/4, μ / 10 si > 3/4."""

    def __init__(self, residuals, damping=1e-8, min_damping=1e-12, max_tries=15):
        self.damping, self.min_damping, self.max_tries = damping, min_damping, max_tries

        @jax.jit
        def prepare(theta):
            r = residuals(theta)
            jac = jax.jacfwd(residuals)(theta)
            return jnp.sum(r ** 2), jac.T @ jac, jac.T @ r

        @jax.jit
        def trial(theta, gram, g, mu):
            scale = mu * jnp.max(jnp.diag(gram))
            chol = jnp.linalg.cholesky(gram + scale * jnp.eye(len(g)))
            d = jax.scipy.linalg.cho_solve((chol, True), g)
            predicted = 2.0 * g @ d - d @ gram @ d
            new = theta - d
            return new, jnp.sum(residuals(new) ** 2), predicted

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
def crossings(x, rho):
    s = np.flatnonzero(np.diff(np.sign(rho - 0.5)))
    return x[s] + (0.5 - rho[s]) * (x[s + 1] - x[s]) / (rho[s + 1] - rho[s])


def evaluate(pb: Problem, theta, unravel):
    pu, pr = unravel(theta)
    rho = np.asarray(jax.vmap(rho_fn, (None, 0))(pr, jnp.asarray(pb.mid)))
    u_nodes = np.asarray(jax.vmap(u_fn, (None, 0))(pu, jnp.asarray(pb.edges)))
    u_ex, _ = pb.exact(rho)
    found = crossings(pb.mid, rho)
    pos = float(np.max(np.abs(found - pb.interfaces))) if len(found) == len(pb.interfaces) else np.nan
    return {
        "rho": rho, "u": u_nodes, "u_exact": u_ex,
        "J_exact": pb.l2sq(u_ex - pb.u_star_nodes) / pb.j_ref,
        "u_error": float(np.sqrt(pb.l2sq(u_nodes - u_ex) / pb.l2sq(u_ex))),
        "rho_error": float(np.sqrt(np.mean((rho - pb.rho_star) ** 2))),
        "interfaces": len(found), "position_error": pos,
    }


# --- test de capacité ---
def capacity(pb: Problem, a, key, log):
    """rho : moindres carrés sur rho_eps ; u : forme forte à rho = rho_eps fixé. Levenberg-Marquardt."""
    x, w = gauss_composite(a.elements, a.gauss)
    xj, sw = jnp.asarray(x), jnp.sqrt(jnp.asarray(w))
    target = jnp.asarray(pb.profile(x))
    k1, k2 = jax.random.split(key)
    theta_r, unr = ravel_pytree(init_rho(k1, a.width, a.depth, pb.volume))
    theta_u, unu = ravel_pytree(init_mlp(k2, a.width, a.depth))

    def res_rho(t):
        return sw * (jax.vmap(rho_fn, (None, 0))(unr(t), xj) - target)

    def res_u(t):
        pu = unu(t)

        def point(y):
            flux = lambda z: pb.conductivity(pb.profile(z, jnp)) * jax.grad(u_fn, 1)(pu, z)  # noqa: E731
            return jax.grad(flux)(y)
        return sw * (-jax.vmap(point)(xj) - pb.f)

    rho_eps = pb.profile(pb.mid)
    h1 = lambda v, dv: pb.l2sq(v) + float(np.sum(dv ** 2) / FINE)  # noqa: E731

    def rho_error(t):
        rho = np.asarray(jax.vmap(rho_fn, (None, 0))(unr(t), jnp.asarray(pb.mid)))
        return float(np.sqrt(np.mean((rho - rho_eps) ** 2) / np.mean(rho_eps ** 2))), rho

    def u_error(t):
        pu = unu(t)
        u = np.asarray(jax.vmap(u_fn, (None, 0))(pu, jnp.asarray(pb.edges)))
        du = np.asarray(jax.vmap(jax.grad(u_fn, 1), (None, 0))(pu, jnp.asarray(pb.mid)))
        return float(np.sqrt(h1(u - pb.u_eps_nodes, du - pb.du_eps) / h1(pb.u_eps_nodes, pb.du_eps))), u

    out = {}
    for name, res, theta, error, tol in (("rho", res_rho, theta_r, rho_error, CAP_RHO_TOL),
                                         ("u", res_u, theta_u, u_error, CAP_U_TOL)):
        lm = LevenbergMarquardt(res)
        t0 = time.time()
        for k in range(a.capacity_iters):   # arrêt dès que l'erreur passe sous tol / 10 (contrôle tous les 20 pas)
            theta, ok = lm.step(theta)
            if not ok or ((k + 1) % 20 == 0 and error(theta)[0] < 0.1 * tol):
                break
        out[name] = (theta, time.time() - t0, k + 1)
    (rho_err, rho_fit), (u_err, u_fit) = rho_error(out["rho"][0]), u_error(out["u"][0])
    res = {
        "rho_error": rho_err, "rho_interfaces": len(crossings(pb.mid, rho_fit)), "u_error_h1": u_err,
        "steps": [out["rho"][2], out["u"][2]], "time_s": out["rho"][1] + out["u"][1],
    }
    res["ok"] = bool(res["rho_error"] < CAP_RHO_TOL and res["rho_interfaces"] == len(pb.interfaces)
                     and res["u_error_h1"] < CAP_U_TOL)
    log(f"capacité ({res['time_s']:.1f} s, {res['steps'][0]} + {res['steps'][1]} pas) : "
        f"rho erreur L2 rel {res['rho_error']:.2e}, "
        f"{res['rho_interfaces']}/{len(pb.interfaces)} interfaces ; u (forme forte, rho_eps fixé) erreur H1 rel "
        f"{res['u_error_h1']:.2e} -> {'OK' if res['ok'] else 'HORS DE PORTÉE'}")
    k = PLOT_STEP
    data = {"x": pb.mid[::k], "rho_star": pb.rho_star[::k], "rho_eps": rho_eps[::k], "rho_fit": rho_fit[::k],
            "xu": pb.edges[::k], "u_eps": pb.u_eps_nodes[::k], "u_fit": u_fit[::k],
            "rho_error": res["rho_error"], "u_error_h1": res["u_error_h1"],
            "title": f"test de capacité : {a.width} x {a.depth} tanh, ε = {pb.eps:g}, {len(pb.interfaces)} "
                     f"interface(s) -> {'OK' if res['ok'] else 'HORS DE PORTÉE'}"}
    return res, data


# --- données des figures (plot1d.py) ---
def save_npz(path: Path, data: dict):
    """Écriture atomique : un lecteur (plot1d.py --watch, copie distante) ne voit jamais un fichier partiel."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **data)
    os.replace(tmp, path)


# --- programme ---
class _Formatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=_Formatter)
    g = p.add_argument_group("cible (ρ* binaire, u* exact)")
    g.add_argument("--n-interfaces", type=int, default=1, metavar="N",
                   help="N interfaces en 0.1 + 0.8 (k + 0.6) / N (jamais en 0.5)")
    g.add_argument("--interfaces", type=float, nargs="+", default=None, metavar="X",
                   help="positions explicites des interfaces (remplace --n-interfaces)")
    g.add_argument("--first-phase", type=int, default=1, choices=(0, 1), help="phase de ρ* en x = 0")
    g = p.add_argument_group("physique : -(A(ρ) u')' = f, A = a2 + (a1 - a2) ρ / (1 + q (1 - ρ))")
    g.add_argument("--a1", type=float, default=1.0, help="conductivité de la phase ρ = 1")
    g.add_argument("--a2", type=float, default=0.01, help="conductivité de la phase ρ = 0")
    g.add_argument("--q", type=float, default=99.0, help="RAMP ; a1/a2 - 1 = loi harmonique (gris non favorisé)")
    g.add_argument("--source", type=float, default=1.0, help="f")
    g = p.add_argument_group("loss : J/J_ref + β résidu + λ Van der Waals + μ volume")
    g.add_argument("--eps", type=float, default=1e-2, help="ε, largeur d'interface (≈ 4.4 ε de 10 à 90 %%)")
    g.add_argument("--lam", type=float, default=1e-3, help="λ, Van der Waals")
    g.add_argument("--beta", type=float, default=1e2, help="β, résidu de l'EDP (forme forte)")
    g.add_argument("--mu", type=float, default=0.0, help="μ, volume (redondant pour ce problème inverse)")
    g = p.add_argument_group("discrétisation")
    g.add_argument("--elements", type=int, default=100, help="éléments de la quadrature de Gauss composite")
    g.add_argument("--gauss", type=int, default=10, help="points de Gauss par élément")
    g.add_argument("--width", type=int, default=20, help="neurones par couche (réseaux u et ρ)")
    g.add_argument("--depth", type=int, default=3, help="couches cachées tanh")
    g = p.add_argument_group("optimisation")
    g.add_argument("--opt", nargs="+", default=["adam:2000", "lbfgs:2000"], metavar="NOM:ITER",
                   help="phases enchaînées ; NOM = adam, lbfgs ou ng (Gauss-Newton amorti)")
    g.add_argument("--lr", type=float, default=1e-3, help="pas d'Adam")
    g.add_argument("--lbfgs-memory", type=int, default=50, help="historique de L-BFGS")
    g.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("test de capacité (avant l'entraînement)")
    g.add_argument("--capacity-only", action="store_true", help="faire le test puis s'arrêter")
    g.add_argument("--no-capacity", action="store_true", help="sauter le test")
    g.add_argument("--capacity-iters", type=int, default=300, help="pas de Gauss-Newton maximum par réseau")
    g = p.add_argument_group("sorties")
    g.add_argument("--name", default="", help="nom du run (défaut : n<nombre d'interfaces>)")
    g.add_argument("--out", default="runs1d", help="dossier des runs")
    g.add_argument("--plot-every", type=int, default=200,
                   help="itérations entre deux mises à jour de live.png (ng : divisé par 20)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    pb = Problem(a)
    name = a.name or f"n{len(pb.interfaces)}"
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

    log(f"run : {run_dir}   appareil : {jax.devices()[0]}")
    log(f"suivi : python plot1d.py {run_dir} --watch"
        + ("" if plot1d is None else f"   (ou l'image {run_dir / 'live.png'})"))
    log(f"cible : interfaces {np.round(pb.interfaces, 4).tolist()}, V* = {pb.volume:.4f}, J_ref = {pb.j_ref:.4e}")
    key = jax.random.PRNGKey(a.seed)
    metrics = {}
    if not a.no_capacity:
        log("test de capacité : ajustement des réseaux sur la cible (environ 1 min sur CPU)")
        cap, data = capacity(pb, a, jax.random.fold_in(key, 1), log)
        save_npz(run_dir / "capacity.npz", data)
        if plot1d is not None:
            plot1d.draw_capacity(data, run_dir / "capacity.png")
        metrics["capacity"] = cap
        if a.capacity_only:
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            return

    ku, kr = jax.random.split(key)
    theta, unravel = ravel_pytree((init_mlp(ku, a.width, a.depth), init_rho(kr, a.width, a.depth, pb.volume)))
    loss = Loss(pb, a, unravel)
    hist = {"terms": [], "phase_starts": []}
    checks = []
    t0 = time.time()
    phase_name = "init"

    def report(done=False, verdict=""):
        snap = evaluate(pb, theta, unravel)
        n = len(hist["terms"])
        checks.append({"iter": n, **{k: snap[k] for k in ("J_exact", "u_error", "rho_error", "interfaces",
                                                         "position_error")}})
        terms = hist["terms"][-1] if n else np.zeros(4)
        if not done:
            log(f"[{phase_name} | it {n:6d}] L {terms.sum():.3e}  J {terms[0]:.2e}  rés {terms[1]:.2e}  "
                f"VdW {terms[2]:.2e}  vol {terms[3]:.1e} | J exact {snap['J_exact']:.2e}  "
                f"erreur u {snap['u_error']:.1e}  "
                f"ρ-ρ* {snap['rho_error']:.2e}  {snap['interfaces']}/{len(pb.interfaces)} interfaces  "
                f"pos {snap['position_error']:.1e}")
        summary = "\n".join([
            f"cible : {len(pb.interfaces)} interface(s) en {np.round(pb.interfaces, 3).tolist()}",
            f"A = {a.a2:g} + {a.a1 - a.a2:g} ρ / (1 + {a.q:g} (1 - ρ)),  f = {a.source:g}",
            f"ε = {a.eps:g}   λ = {a.lam:g}   β = {a.beta:g}   μ = {a.mu:g}",
            f"points : {a.elements} x {a.gauss} Gauss   réseaux : {a.width} x {a.depth} tanh",
            f"optimiseurs : {' '.join(a.opt)}",
            *([f"capacité : {'OK' if metrics['capacity']['ok'] else 'HORS DE PORTÉE'} "
               f"(ρ {metrics['capacity']['rho_error']:.1e}, u H1 {metrics['capacity']['u_error_h1']:.1e})"]
              if "capacity" in metrics else []),
            "",
            f"J PINN / J_ref        {terms[0]:.3e}",
            f"J exact / J_ref       {snap['J_exact']:.3e}",
            f"écart RMS ρ - ρ*      {snap['rho_error']:.3e}",
            f"interfaces            {snap['interfaces']} / {len(pb.interfaces)}",
            f"erreur de position    {snap['position_error']:.2e}   (ε = {a.eps:g})",
            f"erreur relative u     {snap['u_error']:.2e}",
        ])
        title = f"{name}   seed {a.seed}   {phase_name}   it {n}   {time.time() - t0:.0f} s"
        k = PLOT_STEP
        data = {
            "x": pb.mid[::k], "rho_star": pb.rho_star[::k], "rho_eps": pb.profile(pb.mid[::k]), "rho": snap["rho"][::k],
            "xu": pb.edges[::k], "u_star": pb.u_star_nodes[::k], "u_exact": snap["u_exact"][::k], "u": snap["u"][::k],
            "u_error": snap["u_error"], "terms": np.asarray(hist["terms"]).reshape(-1, 4),
            "checks": np.asarray([[c[col] for col in ("iter", "J_exact", "u_error", "rho_error", "interfaces",
                                                      "position_error")] for c in checks], dtype=float),
            "phase_starts": np.asarray(hist["phase_starts"], dtype=int), "n_target": len(pb.interfaces),
            "eps": a.eps, "title": title, "summary": summary, "done": done, "verdict": verdict,
        }
        save_npz(run_dir / "live.npz", data)
        if plot1d is not None:
            fig = plot1d.draw_live(data, run_dir / ("final.png" if done else "live.png"))
            plot1d.plt_close(fig)
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
            lm = LevenbergMarquardt(loss.residuals)
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
    found = snap["interfaces"] == len(pb.interfaces) and snap["position_error"] < a.eps
    verdict = (f"{'TOPOLOGIE RETROUVÉE' if found else 'TOPOLOGIE NON RETROUVÉE'} ({snap['interfaces']}/"
               f"{len(pb.interfaces)} interfaces, erreur de position {snap['position_error']:.2e} pour ε = {a.eps:g}, "
               f"J exact {snap['J_exact']:.3e})")
    checks.pop()   # le dernier état est re-rapporté, marqué comme final
    report(done=True, verdict=verdict)
    metrics["final"] = {**checks[-1], "found": bool(found)}
    metrics["time_s"] = time.time() - t0
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    log(f"terminé en {metrics['time_s']:.0f} s : {verdict}")


if __name__ == "__main__":
    main()
