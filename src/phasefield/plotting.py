"""Figure de suivi, réécrite sous le même nom (écriture dans un fichier temporaire puis renommage)."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LOSS_LABELS = {"total": "totale", "J": "J / J_ref", "residual": "β · résidu",
               "vdw": "λ · Van der Waals", "volume": "μ · volume"}


def _step(ax, edges, rho, **kw):
    ax.stairs(rho, edges, baseline=None, **kw)


def _stage_marks(ax, stage_starts):
    for s in stage_starts[1:]:
        ax.axvline(s, color="0.6", lw=0.6, ls=":")


def draw(path: Path, history: dict, checks: list, snap: dict, refs: dict, title: str, summary: str):
    fig, ax = plt.subplots(2, 3, figsize=(19, 9.5))
    band = refs["band"]
    evals = np.arange(1, len(history["total"]) + 1)

    # rho
    a = ax[0, 0]
    a.axvspan(*band, color="tab:red", alpha=0.08, label="bande de J")
    _step(a, refs["binary"]["edges"], refs["binary"]["rho"], color="k", lw=2,
          label=f"optimum binaire (J/J_ref = {refs['binary']['J']:.4f})")
    _step(a, refs["relaxed"]["edges"], refs["relaxed"]["rho"], color="tab:green", lw=1, ls="--",
          label=f"optimum relaxé, gris permis ({refs['relaxed']['J']:.4f})")
    a.plot(snap["x"], snap["rho"], color="tab:blue", lw=1.8, label="ρ PINN")
    a.set_ylim(-0.05, 1.05)
    a.set_title("ρ(x)")
    a.legend(fontsize=8, loc="center right")

    # u
    a = ax[0, 1]
    a.axvspan(*band, color="tab:red", alpha=0.08)
    a.plot(snap["x"], snap["u_exact"], color="k", lw=2, label="u exacte (ρ du PINN)")
    a.plot(snap["x"], snap["u_pinn"], color="tab:orange", lw=1.5, ls="--", label="u PINN")
    a.set_title(f"u(x)   erreur L2 relative = {snap['u_error']:.2e}")
    a.legend(fontsize=8)

    # loss pondérées
    a = ax[0, 2]
    for k, label in LOSS_LABELS.items():
        a.semilogy(evals, np.maximum(history[k], 1e-16), lw=1.6 if k == "total" else 1.0,
                   color="k" if k == "total" else None, label=label)
    _stage_marks(a, history["stage_starts"])
    a.set_ylim(bottom=max(1e-8, 1e-8 * max(history["total"])))
    a.set_title("termes de la loss (pondérés)")
    a.set_xlabel("évaluations")
    a.legend(fontsize=8, loc="lower left")

    # J
    a = ax[1, 0]
    a.semilogy(evals, history["J"], color="tab:blue", lw=1, label="J PINN / J_ref (dans la loss)")
    if checks:
        ce = [c["eval"] for c in checks]
        a.semilogy(ce, [c["J_exact"] for c in checks], "o", ms=3, color="tab:purple",
                   label="J exact du ρ courant")
        a.semilogy(ce, [c["J_binary"] for c in checks], "s", ms=3, color="tab:red",
                   label="J exact du ρ seuillé à 0.5")
    a.axhline(refs["binary"]["J"], color="k", lw=1.5, label="optimum binaire")
    a.axhline(refs["relaxed"]["J"], color="tab:green", lw=1, ls="--", label="optimum relaxé")
    _stage_marks(a, history["stage_starts"])
    a.set_title("J / J_ref")
    a.set_xlabel("évaluations")
    a.legend(fontsize=8)

    # volume et gris
    a = ax[1, 1]
    a.plot(evals, history["vol"], color="tab:blue", lw=1, label="∫ρ")
    a.axhline(refs["v0"], color="tab:blue", lw=0.8, ls="--", label="V0")
    a.plot(evals, history["grey"], color="tab:gray", lw=1, label="gris G = ∫4ρ(1-ρ)")
    if checks:
        ce = [c["eval"] for c in checks]
        a.plot(ce, [c["vol_binary"] for c in checks], "s", ms=3, color="tab:cyan",
               label="volume du ρ seuillé")
        a.plot(ce, [c["grey_floor"] for c in checks], "_", ms=8, color="k",
               label="plancher 4ε × interfaces")
    _stage_marks(a, history["stage_starts"])
    a.set_ylim(-0.02, 1.02)
    a.set_title("volume et niveau de gris (0 binaire, 1 tout gris)")
    a.set_xlabel("évaluations")
    a.legend(fontsize=8)

    # résumé
    a = ax[1, 2]
    a.axis("off")
    a.text(0.0, 1.0, summary, va="top", ha="left", family="monospace", fontsize=9, transform=a.transAxes)

    for a in ax.flat[:5]:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(tmp, dpi=90)
    plt.close(fig)
    os.replace(tmp, path)   # atomique : le lecteur ne voit jamais une image à moitié écrite
