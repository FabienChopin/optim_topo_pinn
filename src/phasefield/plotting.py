"""Figure de suivi, réécrite sous le même nom (écriture dans un fichier temporaire puis renommage).

1D : courbes ρ(x) et u(x). 2D : cartes de ρ, du design de référence, de u et de l'erreur sur u.
Les panneaux d'historique (loss, J, volume et gris) et le résumé sont communs.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LOSS_LABELS = {"total": "totale", "J": "J / J_ref", "residual": "β · résidu",
               "vdw": "λ · Van der Waals", "volume": "μ · volume"}


def draw(path: Path, history: dict, checks: list, snap: dict, refs: dict, title: str, summary: str):
    two_d = np.ndim(snap["rho"]) == 2
    fig, ax = plt.subplots(2, 4 if two_d else 3, figsize=(24, 10) if two_d else (19, 9.5))
    if two_d:
        _rho_2d(ax[0, 0], snap, refs)
        _ref_2d(ax[0, 1], refs)
        _u_2d(ax[0, 2], ax[0, 3], snap, refs)
        history_axes = ax[1, :3]
        summary_ax = ax[1, 3]
    else:
        _rho_1d(ax[0, 0], snap, refs)
        _u_1d(ax[0, 1], snap, refs)
        history_axes = [ax[0, 2], ax[1, 0], ax[1, 1]]
        summary_ax = ax[1, 2]
    evals = np.arange(1, len(history["total"]) + 1)
    _losses(history_axes[0], history, evals)
    _objective(history_axes[1], history, checks, refs, evals)
    _volume_grey(history_axes[2], history, checks, refs, evals)
    summary_ax.axis("off")
    summary_ax.text(0.0, 1.0, summary, va="top", ha="left", family="monospace", fontsize=9,
                    transform=summary_ax.transAxes)
    for a in history_axes:
        a.grid(alpha=0.3)
        _stage_marks(a, history["stage_starts"])
        a.set_xlabel("évaluations")
    fig.suptitle(title)
    fig.tight_layout()
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(tmp, dpi=90)
    plt.close(fig)
    os.replace(tmp, path)   # atomique : le lecteur ne voit jamais une image à moitié écrite


def _stage_marks(ax, stage_starts):
    for s in stage_starts[1:]:
        ax.axvline(s, color="0.6", lw=0.6, ls=":")


# --- 1D ---
def _rho_1d(a, snap, refs):
    if refs["band"]:
        a.axvspan(*refs["band"], color="tab:red", alpha=0.08, label="bande de J")
    a.stairs(refs["binary"]["rho"], refs["binary"]["edges"], baseline=None, color="k",
             lw=1.2 if "target" in refs else 2, label=f"référence binaire (J/J_ref = {refs['binary']['J']:.4g})")
    a.stairs(refs["relaxed"]["rho"], refs["relaxed"]["edges"], baseline=None, color="tab:green", lw=1, ls="--",
             label=f"optimum relaxé, gris permis ({refs['relaxed']['J']:.4g})")
    if "target" in refs:
        a.stairs(refs["target"]["rho"], refs["target"]["edges"], baseline=None, color="tab:red", lw=1.2,
                 label="ρ* cible (loi harmonique)")
    a.plot(snap["x"], snap["rho"], color="tab:blue", lw=1.8, label="ρ PINN")
    a.set_ylim(-0.05, 1.05)
    a.set_title("ρ(x)")
    a.legend(fontsize=8, loc="center right")
    a.grid(alpha=0.3)


def _u_1d(a, snap, refs):
    if refs["band"]:
        a.axvspan(*refs["band"], color="tab:red", alpha=0.08)
    if snap.get("u_target") is not None:
        a.plot(snap["x"], snap["u_target"], color="tab:red", lw=1, label="u* cible")
    a.plot(snap["x"], snap["u_exact"], color="k", lw=2, label="u exacte (ρ du PINN)")
    a.plot(snap["x"], snap["u_pinn"], color="tab:orange", lw=1.5, ls="--", label="u PINN")
    a.set_title(f"u(x)   erreur L2 relative = {snap['u_error']:.2e}")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)


# --- 2D ---
def _map(a, values, title, **kw):
    im = a.imshow(values.T, origin="lower", extent=[0, 1, 0, 1], **kw)
    plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    a.set_title(title)
    a.set_xlabel("x")
    a.set_ylabel("y")
    return im


def _band_lines(a, refs):
    for b in refs["band"]:
        a.axvline(b, color="tab:red", lw=1, ls="--")


def _contour(a, values, **kw):
    c = (np.arange(values.shape[0]) + 0.5) / values.shape[0]
    a.contour(c, c, values.T, levels=[0.5], **kw)


def _rho_2d(a, snap, refs):
    _map(a, snap["rho"], "ρ PINN (contour : référence binaire)", vmin=0, vmax=1, cmap="gray_r")
    _contour(a, refs["binary"]["rho_grid"], colors="tab:orange", linewidths=1.2)
    _contour(a, snap["rho"], colors="tab:blue", linewidths=0.8)
    _band_lines(a, refs)


def _ref_2d(a, refs):
    _map(a, refs["binary"]["rho_grid"], f"référence binaire (J/J_ref = {refs['binary']['J']:.4f})\n"
         f"référence relaxée : {refs['relaxed']['J']:.4f}", vmin=0, vmax=1, cmap="gray_r")
    _band_lines(a, refs)


def _u_2d(a_u, a_err, snap, refs):
    _map(a_u, snap["u_pinn"], "u PINN", cmap="viridis")
    _band_lines(a_u, refs)
    _map(a_err, np.abs(snap["u_pinn"] - snap["u_exact"]),
         f"|u PINN - u volumes finis|   (L2 relative {snap['u_error']:.2e})", cmap="magma")
    _band_lines(a_err, refs)


# --- historiques ---
def _losses(a, history, evals):
    for k, label in LOSS_LABELS.items():
        a.semilogy(evals, np.maximum(history[k], 1e-16), lw=1.6 if k == "total" else 1.0,
                   color="k" if k == "total" else None, label=label)
    a.set_ylim(bottom=max(1e-8, 1e-8 * max(history["total"])))
    a.set_title("termes de la loss (pondérés)")
    a.legend(fontsize=8, loc="lower left")


def _objective(a, history, checks, refs, evals):
    a.semilogy(evals, history["J"], color="tab:blue", lw=1, label="J PINN / J_ref (dans la loss)")
    if checks:
        ce = [c["eval"] for c in checks]
        a.semilogy(ce, [c["J_exact"] for c in checks], "o", ms=3, color="tab:purple",
                   label="J de référence du ρ courant")
        a.semilogy(ce, [c["J_binary"] for c in checks], "s", ms=3, color="tab:red",
                   label="J de référence du ρ seuillé à 0.5")
    a.axhline(refs["binary"]["J"], color="k", lw=1.5, label="référence binaire")
    if refs["relaxed"]["J"] > 1e-12:   # cible : J relaxé ≈ 0, hors de l'axe logarithmique
        a.axhline(refs["relaxed"]["J"], color="tab:green", lw=1, ls="--", label="référence relaxée")
    a.set_title("J / J_ref")
    a.legend(fontsize=8)


def _volume_grey(a, history, checks, refs, evals):
    a.plot(evals, history["vol"], color="tab:blue", lw=1, label="∫ρ")
    a.axhline(refs["v0"], color="tab:blue", lw=0.8, ls="--", label="V0")
    a.plot(evals, history["grey"], color="tab:gray", lw=1, label="gris G = ∫4ρ(1-ρ)")
    if checks:
        ce = [c["eval"] for c in checks]
        a.plot(ce, [c["vol_binary"] for c in checks], "s", ms=3, color="tab:cyan", label="volume du ρ seuillé")
        a.plot(ce, [c["grey_floor"] for c in checks], "_", ms=8, color="k", label="plancher 4ε × interfaces")
    a.set_ylim(-0.02, 1.02)
    a.set_title("volume et niveau de gris (0 binaire, 1 tout gris)")
    a.legend(fontsize=8)
