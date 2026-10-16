"""Point d'entrée : un run = un dossier runs/<date>_<nom>_s<seed>/ (config, journal, figures, métriques, modèles).

    phasefield-run --config configs/default.toml --set run.seed=4 --set loss.lam=1e-3
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .config import load_config, to_toml
from .models import MLP, DensityNet, StateNet
from .problem import BandProblem1D
from .quadrature import Quadrature
from .references import binary_optimum, relaxed_optimum
from .train import Trainer

log = logging.getLogger("phasefield")
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.toml"


def make_run_dir(runs_dir: str, name: str, seed: int) -> Path:
    run_dir = Path(runs_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}_s{seed}"
    (run_dir / "models").mkdir(parents=True)
    latest = Path(runs_dir) / "latest"            # runs/latest -> dernier run
    tmp = Path(runs_dir) / ".latest.tmp"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(run_dir.name)
    os.replace(tmp, latest)
    return run_dir


def setup_logging(run_dir: Path):
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "log.txt", encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="fichier TOML (défaut : %(default)s)")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.CLE=VALEUR",
                        help="surcharge d'un paramètre (répétable)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)

    torch.set_default_dtype(torch.float64)
    seed_everything(cfg.run.seed)
    run_dir = make_run_dir(cfg.output.runs_dir, cfg.run.name, cfg.run.seed)
    (run_dir / "config.toml").write_text(to_toml(cfg), encoding="utf-8")
    setup_logging(run_dir)
    log.info(f"run : {run_dir}   (suivi : display -update 5 {run_dir / 'live.png'})")
    log.info("surcharges : " + (" ".join(args.set) or "aucune"))

    problem = BandProblem1D(cfg.problem)
    quad = Quadrature(problem.dim, cfg.quadrature.n_elements, cfg.quadrature.n_gauss,
                      problem.lo, problem.hi, cfg.run.device)
    if not all(np.isclose(quad.element_edges, b).any() for b in problem.band):
        log.warning(f"les bornes de la bande {problem.band} ne tombent pas sur des bords d'éléments : "
                    "J est intégré avec une erreur de l'ordre de l'écart entre points")
    log.info(f"J_ref (ρ = V0 partout) = {problem.j_ref:.6e}   points de quadrature : {len(quad)}")

    t = time.time()
    refs = {"band": problem.band, "v0": problem.v0,
            "binary": binary_optimum(problem, seed=cfg.run.seed),
            "relaxed": relaxed_optimum(problem)}
    b, r = refs["binary"], refs["relaxed"]
    log.info(f"optimum binaire : J/J_ref = {b['J']:.4f}, {b['n_interfaces']} interfaces en "
             f"{np.round(b['edges'][1:-1], 4).tolist()}, ρ = {b['rho'].astype(int).tolist()}")
    log.info(f"optimum relaxé (gris permis) : J/J_ref = {r['J']:.4f}, gris {r['grey']:.3f}   "
             f"({time.time() - t:.1f} s)")

    def mlp():
        return MLP(problem.dim, cfg.model.width, cfg.model.depth, problem.lo, problem.hi)

    u_net = StateNet(mlp(), problem.boundary_factor).to(cfg.run.device)
    rho_net = DensityNet(mlp(), problem.v0).to(cfg.run.device)

    trainer = Trainer(cfg, problem, u_net, rho_net, quad, refs, run_dir)
    status = "terminé"
    try:
        trainer.run()
    except KeyboardInterrupt:
        status = "interrompu"
        log.warning("interrompu : sauvegarde de l'état courant")
    trainer.plot()
    shutil.copy(run_dir / "live.png", run_dir / "final.png")
    torch.save({"u": u_net.state_dict(), "rho": rho_net.state_dict(), "eps": trainer.eps},
               run_dir / "models" / "final.pt")
    metrics = {
        "status": status, "time_s": time.time() - trainer.t0, "j_ref": problem.j_ref,
        "binary_optimum": {"J": b["J"], "interfaces": b["edges"][1:-1].tolist(), "rho": b["rho"].tolist()},
        "relaxed_optimum": {"J": r["J"], "grey": r["grey"]},
        "stages": trainer.stages,
        "final": trainer.stages[-1] if trainer.stages else None,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info(f"{status} en {metrics['time_s']:.0f} s   -> {run_dir}")


if __name__ == "__main__":
    main()
