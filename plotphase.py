"""Figures d'un run de phasefield.py, n'importe où : il suffit du fichier live.npz du run.

    python plotphase.py runs/latest                  # écrit live.png dans le dossier
    python plotphase.py runs/latest --watch          # fenêtre redessinée à chaque nouvel état, jusqu'à la fin du run
    python plotphase.py runs/latest --watch --png    # sans affichage graphique : réécrit live.png

Le run peut tourner ailleurs (container, machine distante) : il suffit que live.npz arrive dans le dossier lu,
par n'importe quelle copie ; --watch redessine à chaque nouvelle version.

Dépendances : numpy, matplotlib.
"""

import argparse
import os
import time
import zipfile
from pathlib import Path

import matplotlib
import numpy as np

TERM_NAMES = ("J/J_ref", "β·résidu", "λ·Van der Waals", "μ·volume")


def use_files():
    """Pas de fenêtre : figures écrites dans des fichiers (à appeler avant tout tracé)."""
    matplotlib.use("Agg")


def plt_close(fig):
    import matplotlib.pyplot as plt

    plt.close(fig)


def _save(fig, path: Path):
    tmp = path.with_name(path.stem + ".tmp.png")
    fig.savefig(tmp, dpi=80)
    os.replace(tmp, path)   # atomique : un lecteur ne voit jamais une image à moitié écrite


def _field(fig, a, f, title, cmap="viridis", symmetric=False, vmin=None, vmax=None):
    """Champ f[i, j] en (x_i, y_j) sur [0,1]²."""
    if symmetric:
        vmax = float(np.max(np.abs(f))) or 1.0
        vmin = -vmax
    im = a.imshow(np.asarray(f, dtype=float).T, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    a.set_title(title)
    a.set_aspect("equal")


def _band(a, d, vertical=True):
    for b in np.asarray(d["band"]):
        a.axvline(b, color="C2", ls=":", lw=0.8)


def _history(ax, d):
    """Loss, J, volume et gris au fil des itérations (communs à 1D et 2D)."""
    s = np.asarray(d["stats"]).reshape(-1, 6)
    c = np.asarray(d["checks"]).reshape(-1, 5)   # itération, J exact, J seuillé, volume seuillé, plancher de gris
    it = np.arange(1, len(s) + 1)
    a = ax[0]
    if len(s):
        a.semilogy(it, np.maximum(s[:, :4].sum(axis=1), 1e-16), "k", lw=1.5, label="totale")
        for k, name in enumerate(TERM_NAMES):
            if s[:, k].max() > 0:
                a.semilogy(it, np.maximum(s[:, k], 1e-16), lw=1.0, label=name)
    a.set_title("termes de la loss")
    a = ax[1]
    if len(s):
        a.plot(it, s[:, 0], "C0", lw=0.8, label="J PINN")
    a.plot(c[:, 0], c[:, 1], "C4o-", ms=3, label="J exact (ρ du réseau)")
    a.plot(c[:, 0], c[:, 2], "C3s-", ms=3, label="J exact (ρ seuillé à ½)")
    if "ref_binary_J" in d:
        a.axhline(float(d["ref_binary_J"]), color="k", ls="--", lw=0.8, label="référence binaire")
        a.axhline(float(d["ref_relaxed_J"]), color="0.5", ls=":", lw=0.8, label="référence relaxée")
    vals = np.concatenate([c[:, 1:3].ravel(), s[:, 0] if len(s) else []])
    vals = vals[np.isfinite(vals)]
    if len(vals):
        a.set_ylim(0.0, min(1.5, 1.05 * vals.max()))
    a.set_title("J / J_ref")
    a = ax[2]
    if len(s):
        a.plot(it, s[:, 4], "C0", lw=0.9, label="∫ρ")
        a.plot(it, s[:, 5], "C1", lw=0.9, label="gris ∫4ρ(1-ρ)")
    a.plot(c[:, 0], c[:, 3], "C0s", ms=3, label="volume seuillé")
    a.plot(c[:, 0], c[:, 4], "C1o", ms=3, label="plancher 4ε·interfaces")
    a.axhline(float(d["v0"]), color="C0", ls=":", lw=0.8)
    a.set_title("volume et gris")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
        for st in np.atleast_1d(d["stage_starts"])[1:]:
            a.axvline(st, color="0.6", ls=":", lw=0.7)
        a.set_xlabel("itérations")


def draw(d, path: Path | None = None, fig=None):
    """Figure de suivi à partir des données de live.npz (ou du même dictionnaire) ; rend la figure."""
    import matplotlib.pyplot as plt

    dim = int(d["dim"])
    if fig is None:
        fig = plt.figure(figsize=(19, 9.5) if dim == 1 else (22, 10))
    fig.clf()
    if dim == 1:
        ax = fig.subplots(2, 3)
        x = np.asarray(d["x"])
        a = ax[0, 0]
        if "ref_binary_rho" in d:
            a.stairs(np.asarray(d["ref_binary_rho"]), np.asarray(d["ref_binary_edges"]), color="k", lw=0.9,
                     label="optimum binaire")
            a.plot(np.asarray(d["ref_relaxed_x"]), np.asarray(d["ref_relaxed_rho"]), color="0.5", ls="--", lw=0.8,
                   label="optimum relaxé")
        a.plot(x, d["rho"], "C0", label="ρ réseau")
        _band(a, d)
        a.set_ylim(-0.05, 1.05)
        a.set_title("ρ(x)   (pointillés verts : bande)")
        a = ax[0, 1]
        a.plot(x, d["u_exact"], "k", label="u exacte (ρ du réseau)")
        a.plot(x, d["u_pinn"], "C1--", label="u PINN")
        _band(a, d)
        a.set_title("u(x)")
        for a in (ax[0, 0], ax[0, 1]):
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
        _history([ax[0, 2], ax[1, 0], ax[1, 1]], d)
        text = ax[1, 2]
    else:
        ax = fig.subplots(2, 4)
        rho = np.asarray(d["rho"], dtype=float)
        n = rho.shape[0]
        t = (np.arange(n) + 0.5) / n
        _field(fig, ax[0, 0], rho, "ρ réseau (orange : référence binaire)", cmap="gray_r", vmin=0.0, vmax=1.0)
        if "ref_binary_rho" in d:
            ref = np.asarray(d["ref_binary_rho"], dtype=float)
            if ref.min() < 0.5 < ref.max():
                ax[0, 0].contour(t, t, ref.T, levels=[0.5], colors="C1", linewidths=1.0)
            _field(fig, ax[0, 1], ref, f"référence binaire   J/J_ref = {float(d['ref_binary_J']):.4f}",
                   cmap="gray_r", vmin=0.0, vmax=1.0)
        else:
            ax[0, 1].axis("off")
        _field(fig, ax[0, 2], d["u_pinn"], "u PINN")
        _field(fig, ax[0, 3], np.asarray(d["u_pinn"], dtype=float) - np.asarray(d["u_exact"], dtype=float),
               "u PINN - u volumes finis", cmap="RdBu_r", symmetric=True)
        for a in (ax[0, 0], ax[0, 1], ax[0, 2]):
            _band(a, d)
        _history([ax[1, 0], ax[1, 1], ax[1, 2]], d)
        text = ax[1, 3]
    text.axis("off")
    text.text(0.0, 1.0, str(d["summary"]), va="top", family="monospace", fontsize=9, transform=text.transAxes)
    fig.suptitle(str(d["title"]))
    fig.tight_layout()
    if path is not None:
        _save(fig, path)
    return fig


def _load(path: Path):
    """Contenu de path, ou None s'il manque ou est incomplet (copie en cours)."""
    try:
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="dossier contenant live.npz")
    p.add_argument("--watch", action="store_true", help="suivre le run jusqu'à sa fin")
    p.add_argument("--png", action="store_true", help="avec --watch : écrire live.png au lieu d'ouvrir une fenêtre")
    p.add_argument("--every", type=float, default=3.0, help="secondes entre deux lectures (--watch)")
    a = p.parse_args(argv)
    window = a.watch and not a.png
    if not window:
        use_files()
    import matplotlib.pyplot as plt

    live = a.run_dir / "live.npz"
    seen, fig = None, None
    while True:
        if live.exists() and live.stat().st_mtime != seen:
            d = _load(live)
            if d is not None:
                seen = live.stat().st_mtime
                fig = draw(d, None if window else a.run_dir / "live.png", fig)
                print(str(d["title"]), flush=True)
                if bool(d["done"]):
                    _save(fig, a.run_dir / "final.png")
                    print(f"{d['status']}   -> {a.run_dir / 'final.png'}", flush=True)
                    if window:
                        plt.show()
                    return
        if not a.watch:
            if seen is None:
                raise SystemExit(f"rien à tracer : {live} absent")
            return
        if window and fig is not None:
            plt.pause(a.every)
        else:
            time.sleep(a.every)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
