"""Entraînement : paliers (ε, q) du calendrier de continuation ; Adam au premier palier, puis L-BFGS à chaque palier."""

import logging
import time
from pathlib import Path

import numpy as np
import torch

from . import plotting
from .config import Config
from .diagnostics import evaluate
from .losses import TERMS, compute_losses
from .quadrature import PointSet

log = logging.getLogger("phasefield")


class Trainer:
    def __init__(self, cfg: Config, problem, u_net, rho_net, quad, refs: dict, run_dir: Path):
        self.cfg, self.problem, self.quad, self.refs = cfg, problem, quad, refs
        self.u_net, self.rho_net = u_net, rho_net
        self.params = list(u_net.parameters()) + list(rho_net.parameters())
        self.run_dir = run_dir
        self.history = {k: [] for k in ("total", *TERMS, "local", "vol", "grey", "eps")}
        self.local = None    # déshomogénéisation : (points fixes, matrice du filtre, F[rho_rel]) une fois rho_rel figé
        self.history["stage_starts"] = []
        self.checks = []     # diagnostics exacts aux mises à jour de la figure
        self.stages = []     # bilan de fin de palier
        self.schedule = cfg.continuation.schedule()
        self.stage, (self.eps, problem.q) = 0, self.schedule[0]
        self.t0 = time.time()

    # --- une évaluation de la loss : enregistrement, journal, figure ---
    def loss(self):
        total, info = compute_losses(self.problem, self.u_net, self.rho_net, self.quad, self.eps, self.cfg.loss)
        info["local"] = 0.0
        if self.local is not None:   # sur ses points fixes, y compris pendant la nucléation (points tirés au hasard)
            quad, kernel, target = self.local
            local = self.cfg.dehomogenization.gamma * quad.integrate((kernel @ self.rho_net(quad.points) - target) ** 2)
            total = total + local
            info["local"] = local.item()
            info["total"] += info["local"]
        for k in ("total", *TERMS, "local", "vol", "grey"):
            self.history[k].append(info[k])
        self.history["eps"].append(self.eps)
        n = len(self.history["total"])
        out = self.cfg.output
        if n % out.log_every == 0:
            log.info(f"[palier {self.stage} | éval {n:6d}] total {info['total']:.4e}  J/Jref {info['J']:.4e}  "
                     f"β·rés {info['residual']:.2e}  λ·VdW {info['vdw']:.2e}  μ·vol {info['volume']:.2e}  "
                     f"γ·loc {info['local']:.2e}  "
                     f"vol {info['vol']:.4f}  gris {info['grey']:.3f}")
        if n % out.plot_every == 0:
            self.plot()
        return total

    def snapshot(self) -> dict:
        snap = evaluate(self.problem, self.u_net, self.rho_net, self.cfg.output, self.eps)
        self.checks.append({"eval": len(self.history["total"]),
                            **{k: snap[k] for k in ("J_exact", "J_binary", "vol_binary", "grey_floor")}})
        return snap

    def plot(self, path: Path | None = None) -> dict:
        snap = self.snapshot()
        h = self.history
        c = self.cfg
        title = (f"{c.run.name}   seed {c.run.seed}   palier {self.stage + 1}/{len(self.schedule)}   "
                 f"ε = {self.eps:.2e}   q = {self.problem.q:g}   éval {len(h['total'])}   "
                 f"{time.time() - self.t0:.0f} s")
        summary = "\n".join([
            f"A = {c.problem.a2:g} + {c.problem.a1 - c.problem.a2:g} ρ / (1 + q (1 - ρ))   V0 = {c.problem.v0:g}",
            c.continuation.describe()[0],
            f"λ = {c.loss.lam:g}   β = {c.loss.beta:g}   μ = {c.loss.mu:g}",
            c.continuation.describe()[1],
            f"points : {len(self.quad)} ({c.quadrature.n_elements} x {c.quadrature.n_gauss} Gauss"
            f"{' par direction' if self.problem.dim > 1 else ''})",
            f"réseaux : {c.model.width} x {c.model.depth} tanh",
            "",
            f"J PINN / J_ref       {h['J'][-1]:.4f}",
            f"J réf. / J_ref       {snap['J_exact']:.4f}   ({'exacte' if self.problem.dim == 1 else 'volumes finis'})",
            f"J seuillé / J_ref    {snap['J_binary']:.4f}",
            f"référence binaire    {self.refs['binary']['J']:.4f}",
            f"référence relaxée    {self.refs['relaxed']['J']:.4f}",
            "",
            f"∫ρ                   {h['vol'][-1]:.4f}   (V0 = {c.problem.v0:g})",
            f"volume seuillé       {snap['vol_binary']:.4f}",
            f"gris G               {h['grey'][-1]:.4f}",
            f"plancher 4ε·n        {snap['grey_floor']:.4f}   ({snap['interfaces_text']})",
            f"erreur L2 sur u      {snap['u_error']:.2e}",
        ])
        plotting.draw(path or self.run_dir / "live.png", h, self.checks, snap, self.refs, title, summary)
        return snap

    def freeze_relaxed(self):
        """Fige rho_rel (fin du palier dehomogenization.stage) et active la contrainte de densité locale."""
        d = self.cfg.dehomogenization
        if self.problem.dim != 1:
            raise ValueError("déshomogénéisation : 1D seulement (filtre en matrice dense)")
        x = self.quad.points
        dist = (x - x.T).abs()
        kernel = torch.clamp(1.0 - dist / d.ell, min=0.0) * self.quad.weights.T   # chapeau de rayon ell
        kernel = kernel / kernel.sum(dim=1, keepdim=True)                          # moyenne locale
        with torch.no_grad():
            rho_rel = self.rho_net(x)
            target = kernel @ rho_rel
        self.local = (self.quad, kernel, target)
        log.info(f"déshomogénéisation : ρ relaxé figé (gris {self.history['grey'][-1]:.3f}), "
                 f"contrainte locale ℓ = {d.ell:g}, γ = {d.gamma:g}")
        if d.init == "projection":
            # laminé : dans chaque période, une couche de phase 1 centrée, d'épaisseur rho_rel × période
            from .diagnose import FIT_GRID, fit_profile
            _, mid = self.problem.cells(FIT_GRID[1])
            with torch.no_grad():
                r = self.rho_net(torch.tensor(mid.reshape(-1, 1), dtype=x.dtype, device=x.device)).cpu().numpy().ravel()
            phase = (mid / d.init_period) % 1.0
            binary = (np.abs(phase - 0.5) < 0.5 * r).astype(float)
            fit_profile(self.problem, self.rho_net, self.eps, binary, in_place=True)
            log.info(f"déshomogénéisation : ρ recalé sur un laminé de période {d.init_period:g} "
                     f"({int(np.count_nonzero(np.diff(binary)))} interfaces)")

    def nucleate(self):
        """Adam sur des points de Monte-Carlo retirés à chaque pas (voir TrainConfig)."""
        c = self.cfg
        log.info(f"nucléation : {c.train.nucleation_iters} pas d'Adam sur {len(self.quad)} points tirés au hasard")
        fixed = self.quad
        opt = torch.optim.Adam(self.params, lr=c.train.nucleation_lr)
        try:
            for _ in range(c.train.nucleation_iters):
                self.quad = PointSet.monte_carlo(self.problem.dim, len(fixed), self.problem.lo, self.problem.hi,
                                                 fixed.points.device)
                opt.zero_grad()
                self.loss().backward()
                opt.step()
        finally:
            self.quad = fixed

    # --- boucle des paliers ---
    def run(self):
        c = self.cfg
        for stage, (eps, q) in enumerate(self.schedule):
            self.stage, self.eps, self.problem.q = stage, eps, q
            self.history["stage_starts"].append(len(self.history["total"]))
            log.info(f"===== palier {stage + 1}/{len(self.schedule)}   ε = {self.eps:.3e}   "
                     f"q = {self.problem.q:g} =====")
            t_stage = time.time()
            if stage == 0 and c.train.adam_iters > 0:
                opt = torch.optim.Adam(self.params, lr=c.train.adam_lr)
                for _ in range(c.train.adam_iters):
                    opt.zero_grad()
                    self.loss().backward()
                    opt.step()
            if stage + 1 in c.train.nucleation_stages:
                self.nucleate()
            lbfgs = torch.optim.LBFGS(self.params, lr=1.0, max_iter=c.train.lbfgs_iters,
                                      max_eval=int(1.25 * c.train.lbfgs_iters),
                                      history_size=c.train.lbfgs_history, tolerance_grad=1e-12,
                                      tolerance_change=1e-15, line_search_fn="strong_wolfe")

            def closure(lbfgs=lbfgs):
                lbfgs.zero_grad()
                total = self.loss()
                total.backward()
                return total

            lbfgs.step(closure)
            if stage + 1 == c.dehomogenization.stage:
                self.freeze_relaxed()
            snap = self.plot()
            h = self.history
            summary = {"stage": stage, "eps": self.eps, "q": self.problem.q, "evals": len(h["total"]),
                       "time_s": time.time() - t_stage,
                       "lbfgs_n_iter": lbfgs.state[self.params[0]].get("n_iter", 0),   # = lbfgs_iters : plafond
                       **{k: h[k][-1] for k in ("total", *TERMS, "vol", "grey")},
                       **{k: snap[k] for k in ("J_exact", "J_binary", "vol_binary", "interfaces",
                                               "grey_floor", "u_error")}}
            self.stages.append(summary)
            log.info(f"fin du palier {stage + 1} ({summary['time_s']:.1f} s) : J PINN {h['J'][-1]:.4f}  "
                     f"J exact {snap['J_exact']:.4f}  J seuillé {snap['J_binary']:.4f} "
                     f"(vol {snap['vol_binary']:.3f})  gris {h['grey'][-1]:.3f} (plancher {snap['grey_floor']:.3f})  "
                     f"{snap['interfaces_text']}  erreur u {snap['u_error']:.2e}")
            torch.save({"u": self.u_net.state_dict(), "rho": self.rho_net.state_dict(), "eps": self.eps,
                        "q": self.problem.q},
                       self.run_dir / "models" / f"stage_{stage:02d}.pt")
