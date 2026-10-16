"""Entraînement : paliers géométriques en eps ; Adam au premier palier, puis L-BFGS à chaque palier."""

import logging
import time
from pathlib import Path

import torch

from . import plotting
from .config import Config
from .diagnostics import evaluate
from .losses import TERMS, compute_losses

log = logging.getLogger("phasefield")


class Trainer:
    def __init__(self, cfg: Config, problem, u_net, rho_net, quad, refs: dict, run_dir: Path):
        self.cfg, self.problem, self.quad, self.refs = cfg, problem, quad, refs
        self.u_net, self.rho_net = u_net, rho_net
        self.params = list(u_net.parameters()) + list(rho_net.parameters())
        self.run_dir = run_dir
        self.history = {k: [] for k in ("total", *TERMS, "vol", "grey", "eps")}
        self.history["stage_starts"] = []
        self.checks = []     # diagnostics exacts aux mises à jour de la figure
        self.stages = []     # bilan de fin de palier
        self.stage, self.eps = 0, cfg.continuation.eps0
        self.t0 = time.time()

    # --- une évaluation de la loss : enregistrement, journal, figure ---
    def loss(self):
        total, info = compute_losses(self.problem, self.u_net, self.rho_net, self.quad, self.eps, self.cfg.loss)
        for k in ("total", *TERMS, "vol", "grey"):
            self.history[k].append(info[k])
        self.history["eps"].append(self.eps)
        n = len(self.history["total"])
        out = self.cfg.output
        if n % out.log_every == 0:
            log.info(f"[palier {self.stage} | éval {n:6d}] total {info['total']:.4e}  J/Jref {info['J']:.4e}  "
                     f"β·rés {info['residual']:.2e}  λ·VdW {info['vdw']:.2e}  μ·vol {info['volume']:.2e}  "
                     f"vol {info['vol']:.4f}  gris {info['grey']:.3f}")
        if n % out.plot_every == 0:
            self.plot()
        return total

    def snapshot(self) -> dict:
        snap = evaluate(self.problem, self.u_net, self.rho_net, self.cfg.output.n_exact, self.eps)
        self.checks.append({"eval": len(self.history["total"]),
                            **{k: snap[k] for k in ("J_exact", "J_binary", "vol_binary", "grey_floor")}})
        return snap

    def plot(self, path: Path | None = None) -> dict:
        snap = self.snapshot()
        h = self.history
        c = self.cfg
        title = (f"{c.run.name}   seed {c.run.seed}   palier {self.stage + 1}/{c.continuation.n_stages}   "
                 f"ε = {self.eps:.2e}   éval {len(h['total'])}   {time.time() - self.t0:.0f} s")
        summary = "\n".join([
            f"A = {c.problem.a1:g} ρ + {c.problem.a2:g} (1 - ρ)   V0 = {c.problem.v0:g}",
            f"λ = {c.loss.lam:g}   β = {c.loss.beta:g}   μ = {c.loss.mu:g}",
            f"ε : {c.continuation.eps0:g} -> {c.continuation.eps_min:g} ({c.continuation.n_stages} paliers)",
            f"points : {len(self.quad)} ({c.quadrature.n_elements} x {c.quadrature.n_gauss} Gauss)",
            f"réseaux : {c.model.width} x {c.model.depth} tanh",
            "",
            f"J PINN / J_ref       {h['J'][-1]:.4f}",
            f"J exact / J_ref      {snap['J_exact']:.4f}",
            f"J seuillé / J_ref    {snap['J_binary']:.4f}",
            f"optimum binaire      {self.refs['binary']['J']:.4f}",
            f"optimum relaxé       {self.refs['relaxed']['J']:.4f}",
            "",
            f"∫ρ                   {h['vol'][-1]:.4f}   (V0 = {c.problem.v0:g})",
            f"volume seuillé       {snap['vol_binary']:.4f}",
            f"gris G               {h['grey'][-1]:.4f}",
            f"plancher 4ε·n        {snap['grey_floor']:.4f}   ({snap['n_interfaces']} interfaces)",
            f"erreur L2 sur u      {snap['u_error']:.2e}",
        ])
        plotting.draw(path or self.run_dir / "live.png", h, self.checks, snap, self.refs, title, summary)
        return snap

    # --- boucle des paliers ---
    def run(self):
        c = self.cfg
        for stage in range(c.continuation.n_stages):
            self.stage, self.eps = stage, c.continuation.eps(stage)
            self.history["stage_starts"].append(len(self.history["total"]))
            log.info(f"===== palier {stage + 1}/{c.continuation.n_stages}   ε = {self.eps:.3e} =====")
            t_stage = time.time()
            if stage == 0 and c.train.adam_iters > 0:
                opt = torch.optim.Adam(self.params, lr=c.train.adam_lr)
                for _ in range(c.train.adam_iters):
                    opt.zero_grad()
                    self.loss().backward()
                    opt.step()
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
            snap = self.plot()
            h = self.history
            summary = {"stage": stage, "eps": self.eps, "evals": len(h["total"]),
                       "time_s": time.time() - t_stage,
                       **{k: h[k][-1] for k in ("total", *TERMS, "vol", "grey")},
                       **{k: snap[k] for k in ("J_exact", "J_binary", "vol_binary", "n_interfaces",
                                               "grey_floor", "u_error")}}
            self.stages.append(summary)
            log.info(f"fin du palier {stage + 1} ({summary['time_s']:.1f} s) : J PINN {h['J'][-1]:.4f}  "
                     f"J exact {snap['J_exact']:.4f}  J seuillé {snap['J_binary']:.4f} "
                     f"(vol {snap['vol_binary']:.3f})  gris {h['grey'][-1]:.3f} (plancher {snap['grey_floor']:.3f})  "
                     f"interfaces {snap['n_interfaces']}  erreur u {snap['u_error']:.2e}")
            torch.save({"u": self.u_net.state_dict(), "rho": self.rho_net.state_dict(), "eps": self.eps},
                       self.run_dir / "models" / f"stage_{stage:02d}.pt")
