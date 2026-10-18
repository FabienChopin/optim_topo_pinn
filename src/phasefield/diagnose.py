"""Diagnostic a posteriori d'un run : qu'est-ce qui a limité le design à chaque palier ?

    python run.py diagnose runs/latest
    python -m phasefield.diagnose runs/latest --stage 7

Recharge config.toml et models/stage_XX.pt, sans réentraîner l'optimisation. Mesures par palier :
- netteté r = 1 / (4 ε |∇ρ|) sur le contour ρ = 0.5 : r = 1 pour le profil optimal sigmoïde(d / ε)
  (d : distance signée à l'interface), r > 1 si la transition est plus large que ce que ε demande ;
- capacité du réseau ρ : une copie du réseau est recalée (moindres carrés) sur sigmoïde(d / ε) pour
  le contour courant. Sa netteté r_fit dit si le réseau sait représenter le profil demandé. Si
  r_fit ≈ 1 alors que r > 1, la transition large est choisie par l'optimisation (gris voulu) ;
- points par interface : largeur 10-90 % demandée (2 ln 9 · ε ≈ 4.4 ε) / écart moyen entre points ;
- écart de quadrature : J, résidu et Van der Waals réintégrés sur une quadrature plus fine
  (le réseau peut tricher entre les points) ;
- gris en excès : gris - 4 ε × interfaces (plancher du profil optimal), gris voulu par la loi matériau ;
- plus petit détail (couche en 1D, épaisseur locale en 2D) / largeur d'interface demandée ;
- L-BFGS arrêté au plafond d'itérations ;
- erreur sur u (solution exacte en 1D, volumes finis en 2D).
Résultat : un tableau dans le terminal et diagnose.json dans le dossier du run.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .config import load_config
from .diagnostics import evaluate, interface_measure
from .losses import compute_losses
from .models import make_nets
from .problem import make_problem
from .quadrature import PointSet, Quadrature
from .references import binary_optimum, laminate_optimum

WIDTH_FACTOR = 2.0 * math.log(9.0)   # largeur 10-90 % de sigmoïde(x / ε), en unités de ε
FINE_GRID = {1: 20000, 2: 600}       # grille de ρ (netteté, gris, détails)
FIT_GRID = {1: 4000, 2: 300}         # grille du test de capacité
FIT_ITERS = 2000
FINE_QUAD = {1: 4, 2: 2}             # facteur sur n_elements pour la quadrature fine

# seuils des verdicts
MIN_POINTS = 4.0          # points de quadrature à travers une interface
MAX_SHARPNESS = 1.5       # r_fit au-delà : le réseau ne sait pas raidir l'interface
MAX_GAP = 0.05            # écart relatif quadrature fine / d'entraînement (J, Van der Waals)
MAX_RESIDUAL_RATIO = 2.0  # résidu fin / résidu d'entraînement
MAX_GREY = 0.02           # gris en excès
MIN_FEATURE = 3.0         # plus petit détail, en largeurs d'interface
MAX_U_ERROR = 1e-2


def integrals(problem, u_net, rho_net, quad: Quadrature, eps: float, weights, batch: int = 20000) -> dict:
    """Termes de la loss (sauf volume, non linéaire), ∫ρ et gris, intégrés par tranches."""
    keys = ("J", "residual", "vdw", "vol", "grey")
    out = dict.fromkeys(keys, 0.0)
    for k in range(0, len(quad), batch):
        chunk = PointSet(quad.points[k:k + batch], quad.weights[k:k + batch])
        _, info = compute_losses(problem, u_net, rho_net, chunk, eps, weights)
        for key in keys:
            out[key] += info[key]
    return out


def _rel_gap(fine: float, coarse: float) -> float:
    return abs(fine - coarse) / max(abs(fine), 1e-30)


def smallest_feature(binary: np.ndarray, width: float) -> float:
    """1D : plus courte couche. 2D : plus petite épaisseur de composante connexe, 2 × max de la
    distance à l'autre phase dans la composante ; les composantes d'aire < width² (confettis du
    seuillage d'un ρ gris proche de 0.5) sont ignorées. nan si une seule phase."""
    n = binary.shape[0]
    if binary.min() == binary.max():
        return math.nan
    if binary.ndim == 1:
        cuts = np.flatnonzero(np.diff(binary)) + 1
        return float(np.diff(np.r_[0, cuts, n]).min()) / n
    thickness = []
    for phase in (0.0, 1.0):
        mask = binary == phase
        dist = ndimage.distance_transform_edt(mask)
        labels, n_comp = ndimage.label(mask)
        ids = np.arange(1, n_comp + 1)
        big = np.bincount(labels.ravel(), minlength=n_comp + 1)[1:] >= (width * n) ** 2
        thickness += list(2.0 * np.asarray(ndimage.maximum(dist, labels, ids))[big] / n)
    return float(min(thickness)) if thickness else math.nan


def _rho_grid(problem, rho_net, n: int) -> np.ndarray:
    """ρ aux centres de n (1D) ou n x n (2D) cellules."""
    pts = problem.cells(n)[1].reshape(-1, problem.dim)
    x = torch.tensor(pts, dtype=torch.get_default_dtype(), device=next(rho_net.parameters()).device)
    with torch.no_grad():
        rho = torch.cat([rho_net(x[k:k + 65536]) for k in range(0, len(x), 65536)])
    return rho.cpu().numpy().astype(float).reshape((n,) * problem.dim)


def sharpness(rho: np.ndarray, eps: float) -> float:
    """r = 1 / (4 ε |∇ρ|), médiane sur les cellules où ρ est proche de 0.5 ; nan sans interface."""
    grad = np.gradient(rho, 1.0 / rho.shape[0])
    slope = np.abs(grad) if rho.ndim == 1 else np.hypot(*grad)
    contour = np.abs(rho - 0.5) < 0.1
    return float(1.0 / (4.0 * eps * np.median(slope[contour]))) if contour.any() else math.nan


def signed_distance(binary: np.ndarray) -> np.ndarray:
    """Distance signée au contour ρ = 0.5 (> 0 dans la phase 1), aux centres des cellules."""
    n = binary.shape[0]
    inside = binary > 0.5
    if binary.ndim == 1:
        x = (np.arange(n) + 0.5) / n
        cuts = x[:-1][np.diff(binary) != 0] + 0.5 / n
        dist = np.abs(x[:, None] - cuts[None, :]).min(axis=1)
        return np.where(inside, dist, -dist)
    d_in = ndimage.distance_transform_edt(inside) - 0.5
    d_out = ndimage.distance_transform_edt(~inside) - 0.5
    return np.where(inside, d_in, -d_out) / n


def fit_profile(problem, rho_net, eps: float, binary: np.ndarray) -> np.ndarray:
    """ρ (grille FIT_GRID) d'une copie du réseau ρ recalée sur sigmoïde(d / ε), d distance signée au
    contour de binary (même grille)."""
    n = FIT_GRID[problem.dim]
    net = copy.deepcopy(rho_net)
    device = next(net.parameters()).device
    dtype = torch.get_default_dtype()
    x = torch.tensor(problem.cells(n)[1].reshape(-1, problem.dim), dtype=dtype, device=device)
    target = torch.sigmoid(torch.tensor(signed_distance(binary).reshape(-1, 1) / eps, dtype=dtype, device=device))
    opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=FIT_ITERS, history_size=50,
                            tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ((net(x) - target) ** 2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return _rho_grid(problem, net, n)


def fitted_sharpness(problem, rho_net, eps: float) -> float:
    """Netteté atteinte par le réseau recalé sur le profil optimal du contour courant."""
    binary = (_rho_grid(problem, rho_net, FIT_GRID[problem.dim]) > 0.5).astype(float)
    if binary.min() == binary.max():
        return math.nan
    return sharpness(fit_profile(problem, rho_net, eps, binary), eps)


def reference_capacity(problem, rho_net, eps: float, ref: dict) -> dict:
    """1D : le réseau, recalé sur le profil optimal du design binaire de référence, en reproduit-il les
    interfaces ? (le contour courant peut être bien plus simple que la référence)"""
    edges, mid = problem.cells(FIT_GRID[1])
    binary = ref["rho"][np.clip(np.searchsorted(ref["edges"], mid, side="right") - 1, 0, len(ref["rho"]) - 1)]
    fit = fit_profile(problem, rho_net, eps, binary)
    return {"ref_interfaces": interface_measure(binary), "ref_interfaces_fit": interface_measure(fit > 0.5),
            "ref_sharpness_fit": sharpness(fit, eps)}


def rho_measures(problem, rho_net, eps: float) -> dict:
    """Netteté, gris et détails de ρ sur une grille fine."""
    rho = _rho_grid(problem, rho_net, FINE_GRID[problem.dim])
    binary = (rho > 0.5).astype(float)
    interfaces = interface_measure(binary)
    grey = float(np.mean(4.0 * rho * (1.0 - rho)))
    return {"sharpness": sharpness(rho, eps), "sharpness_fit": fitted_sharpness(problem, rho_net, eps),
            "interfaces": interfaces, "grey": grey, "grey_excess": grey - 4.0 * eps * interfaces,
            "width": WIDTH_FACTOR * eps, "smallest_feature": smallest_feature(binary, WIDTH_FACTOR * eps)}


def lbfgs_capped(cfg, stages: list, stage: int) -> bool | None:
    """True si L-BFGS s'est arrêté au plafond d'itérations, None si on ne peut pas le savoir.
    Anciens runs (sans lbfgs_n_iter) : chaque itération coûte au moins une évaluation et
    max_eval = 1.25 lbfgs_iters, donc on ne conclut qu'aux deux extrêmes."""
    if stage >= len(stages):
        return None
    s = stages[stage]
    iters = cfg.train.lbfgs_iters
    if "lbfgs_n_iter" in s:
        return s["lbfgs_n_iter"] >= iters
    evals = s["evals"] - (stages[stage - 1]["evals"] if stage > 0 else cfg.train.adam_iters)
    if evals < iters:
        return False
    return True if evals >= int(1.25 * iters) else None


def verdict(d: dict) -> list[str]:
    limits = []
    points_ok = d["points_per_interface"] >= MIN_POINTS and d["residual_ratio"] <= MAX_RESIDUAL_RATIO \
        and d["vdw_gap"] <= MAX_GAP and d["J_gap"] <= MAX_GAP
    if not points_ok:
        limits.append("points")
    ref_missed = "ref_interfaces" in d and (d["ref_interfaces_fit"] != d["ref_interfaces"]
                                            or d["ref_sharpness_fit"] > MAX_SHARPNESS)
    if d["sharpness_fit"] > MAX_SHARPNESS or ref_missed:
        limits.append("réseau ρ")
    elif d["grey_excess"] > MAX_GREY:
        # q < a1/a2 - 1 : le gris est plus conducteur qu'un mélange réel, l'optimum le veut ;
        # au-delà, la loi le pénalise : s'il reste, c'est que l'optimisation ne l'a pas résorbé
        limits.append("loi (q)" if d["q"] < d["q_harmonic"] else "gris résiduel")
    if d["smallest_feature"] < MIN_FEATURE * d["width"]:
        limits.append("ε")
    if d["lbfgs_capped"]:
        limits.append("optimiseur")
    if points_ok and d["u_error"] > MAX_U_ERROR:
        limits.append("réseau u")
    return limits


def diagnose_stage(cfg, problem, quad, fine_quad, u_net, rho_net, checkpoint: Path, stages: list,
                   ref: dict | None) -> dict:
    stage = int(checkpoint.stem.split("_")[1])
    state = torch.load(checkpoint, map_location="cpu")
    u_net.load_state_dict(state["u"])
    rho_net.load_state_dict(state["rho"])
    eps = state["eps"]
    problem.q = state.get("q", cfg.continuation.schedule()[stage][1])   # anciens modèles : q du calendrier

    coarse = integrals(problem, u_net, rho_net, quad, eps, cfg.loss)
    fine = integrals(problem, u_net, rho_net, fine_quad, eps, cfg.loss)
    rho = rho_measures(problem, rho_net, eps)
    snap = evaluate(problem, u_net, rho_net, cfg.output, eps)
    h_q = 1.0 / (cfg.quadrature.n_elements * cfg.quadrature.n_gauss)
    d = {
        "stage": stage, "eps": eps, "q": problem.q, "q_harmonic": problem.a1 / problem.a2 - 1.0,
        **rho,
        "points_per_interface": rho["width"] / h_q,
        "J_gap": _rel_gap(fine["J"], coarse["J"]),
        "vdw_gap": _rel_gap(fine["vdw"], coarse["vdw"]),
        "residual_ratio": fine["residual"] / max(coarse["residual"], 1e-30),
        "lbfgs_capped": lbfgs_capped(cfg, stages, stage),
        "u_error": snap["u_error"], "J_exact": snap["J_exact"], "J_binary": snap["J_binary"],
    }
    if ref is not None:
        d.update(reference_capacity(problem, rho_net, eps, ref))
    d["limits"] = verdict(d)
    return d


def _row(d: dict) -> str:
    capped = {True: "oui", False: "non", None: "?"}[d["lbfgs_capped"]]
    feature = d["smallest_feature"] / d["width"]
    return (f"{d['stage'] + 1:6d}  {d['eps']:8.2e}  {d['q']:6g}  {d['sharpness']:5.2f}  {d['sharpness_fit']:5.2f}  "
            f"{d['points_per_interface']:6.1f}  {d['residual_ratio']:6.2f}  {d['vdw_gap']:6.1%}  {d['J_gap']:6.1%}  "
            f"{d['grey']:5.3f}  {d['grey_excess']:6.3f}  {feature:6.1f}  {capped:>6}  "
            f"{d['u_error']:8.1e}  {_ref_text(d):>9}  {', '.join(d['limits']) or '-'}")


def _ref_text(d: dict) -> str:
    if "ref_interfaces" not in d:
        return "-"
    return f"{d['ref_interfaces_fit']}/{d['ref_interfaces']} {d['ref_sharpness_fit']:.1f}"


HEADER = ("palier         ε       q      r  r_fit  pts/if  rés f/e   ΔVdW      ΔJ   gris   excès  détail  "
          "plafond  erreur u   réf fit  limitant")
LEGEND = f"""
r        netteté : 1 / (4 ε |∇ρ|) sur ρ = 0.5 (1 = profil optimal sigmoïde(d / ε), > 1 : transition large)
r_fit    netteté d'une copie du réseau ρ recalée sur sigmoïde(d / ε) (> {MAX_SHARPNESS} : le réseau limite)
pts/if   points de quadrature à travers une interface (4.4 ε / écart moyen ; < {MIN_POINTS:g} : sous-résolu)
rés f/e  résidu sur quadrature fine / d'entraînement (> {MAX_RESIDUAL_RATIO:g} : le réseau triche entre les points)
ΔVdW ΔJ  écart relatif quadrature fine / d'entraînement (> {MAX_GAP:.0%} : sous-résolu)
excès    gris - plancher 4 ε × interfaces (> {MAX_GREY:g}) : q < a1/a2 - 1, gris voulu par la loi (monter q) ;
         q ≥ a1/a2 - 1, gris pénalisé mais pas résorbé (gris résiduel : plus d'itérations, paliers plus longs)
détail   plus petit détail en largeurs d'interface (< {MIN_FEATURE:g} : ε limite la finesse)
plafond  L-BFGS arrêté au nombre max d'itérations
réf fit  (1D) réseau recalé sur le profil optimal du design binaire de référence : interfaces
         reproduites / interfaces de la référence, puis netteté (le réseau sait-il faire la cible ?)"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="dossier du run (ex. runs/latest)")
    parser.add_argument("--stage", type=int, action="append", help="palier (compté depuis 1), répétable ; "
                        "défaut : tous")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.toml")
    torch.set_default_dtype(getattr(torch, cfg.run.dtype))
    metrics_path = run_dir / "metrics.json"
    stages = json.loads(metrics_path.read_text())["stages"] if metrics_path.exists() else []

    problem = make_problem(cfg)
    q = cfg.quadrature
    quad = Quadrature(problem.dim, q.n_elements, q.n_gauss, problem.lo, problem.hi, args.device)
    fine_quad = Quadrature(problem.dim, q.n_elements * FINE_QUAD[problem.dim], q.n_gauss,
                           problem.lo, problem.hi, args.device)

    cfg.run.device = args.device
    u_net, rho_net = make_nets(cfg, problem)

    ref = None
    if problem.dim == 1:   # référence binaire, recalculée (exacte, quelques secondes)
        ref = (laminate_optimum(problem, cfg.loss.lam) if problem.objective_kind == "target"
               else binary_optimum(problem, seed=cfg.run.seed))

    checkpoints = sorted((run_dir / "models").glob("stage_*.pt"))
    if args.stage:
        checkpoints = [c for c in checkpoints if int(c.stem.split("_")[1]) + 1 in args.stage]
    print(f"{run_dir}   points : {len(quad)} (fin : {len(fine_quad)})   réseaux {cfg.model.width} x {cfg.model.depth}")
    print(HEADER)
    results = []
    for checkpoint in checkpoints:
        d = diagnose_stage(cfg, problem, quad, fine_quad, u_net, rho_net, checkpoint, stages, ref)
        results.append(d)
        print(_row(d), flush=True)
    print(LEGEND)
    (run_dir / "diagnose.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
